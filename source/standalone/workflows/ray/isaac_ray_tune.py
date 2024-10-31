# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import importlib.util
import os
import subprocess
import sys
from time import sleep

import isaac_ray_util
import ray
from ray import air, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.repeater import Repeater

"""
This script breaks down an aggregate tuning job, as defined by a hyperparameter sweep configuration,
into individual jobs (shell commands) to run on the GPU-enabled nodes of the cluster.
By default, (unless combined as a sub-job in a resource-wrapped aggregate job), one worker is created
for each GPU-enabled node in the cluster for each individual job.

Each hyperparameter sweep configuration should include the workflow,
runner arguments, and hydra arguments to vary.

This assumes that all workers in a cluster are homogeneous. For heterogeneous workloads,
create several heterogeneous clusters (with homogeneous nodes in each cluster),
then submit several overall-cluster jobs with :file:`../submit_isaac_ray_job.py`.
KubeRay clusters on Google GKE can be created with :file:`../launch.py`

Usage:

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/workflows/ray/isaac_ray_tune.py -h

    # Examples
    # Local
    ./isaaclab.sh -p source/standalone/workflows/ray/isaac_ray_tune.py --run_mode local \
    --storage_path /home/<USERNAME>/exp_result \
    --cfg_file hyperparameter_tuning/vision_cartpole_cfg.py \
    --cfg_cls CartpoleRGBNoTuneJobCfg
    # Remote (run grok cluster or create config file mentioned in :file:`submit_isaac_ray_job.py`)
    ./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py \
    --aggregate_jobs isaac_ray_tune.py
    --cfg_file hyperparameter_tuning/vision_cartpole_cfg.py \
    --cfg_cls CartpoleRGBNoTuneJobCfg

"""

DOCKER_PREFIX = "/workspace/isaaclab/"
BASE_DIR = os.path.expanduser("~")
PYTHON_EXEC = "./isaaclab.sh -p"
RL_GAMES_WORKFLOW = "source/standalone/workflows/rl_games/train.py"


class IsaacLabTuneTrainable(tune.Trainable):
    """The Isaac Ray Tune Trainable.
    This class uses the standalone workflows to start jobs, along with the hydra integration.
    This class achieves Ray-based logging through reading the tensorboard logs from
    the standalone workflows.
    """

    def setup(self, config: dict) -> None:
        self.data = None

        if hasattr(self, "checkpoint"):
            config["runner_args"]["--checkpoint"] = self.checkpoint
        self.invoke_cmd = isaac_ray_util.get_invocation_command_from_cfg(cfg=config, python_cmd=PYTHON_EXEC)
        print(f"[INFO]: Recovered invocation with {self.invoke_cmd}")
        self.experiment = None

    def reset_config(self, new_config):
        self.setup(new_config)
        return True

    def step(self) -> dict:
        if self.experiment is None:  # start experiment
            # When including this as first step instead of setup, experiments get scheduled faster
            # Don't want to block the scheduler while the experiment spins up
            print(f"[INFO]: Invoking experiment as first step with {self.invoke_cmd}...")
            experiment = isaac_ray_util.execute_job(
                self.invoke_cmd,
                identifier_string="",
                extract_experiment=True,
                persistent_dir=BASE_DIR,
            )
            self.experiment = experiment
            print(f"[INFO]: Tuner recovered experiment info {experiment}")
            self.proc = experiment["proc"]
            self.experiment_name = experiment["experiment_name"]
            self.isaac_logdir = experiment["logdir"]
            self.tensorboard_logdir = self.isaac_logdir + f"/{self.experiment_name}/summaries"
            self.done = False

        if self.proc is None:
            raise ValueError("Could not start trial.")

        if self.proc.poll() is not None:  # process finished, signal finish
            self.data["done"] = True
            print("[INFO]: Process finished, returning...")
            if self.done:
                raise ValueError("Previously tried to halt trial to no avail.")
            self.done = True
            return self.data
        else:  # wait until the logs are ready or fresh
            data = isaac_ray_util.load_tensorboard_logs(self.tensorboard_logdir)

            while data is None:
                data = isaac_ray_util.load_tensorboard_logs(self.tensorboard_logdir)
                sleep(2)  # Lazy report metrics to avoid performance overhead

            if self.data is not None:
                while isaac_ray_util._dicts_equal(data, self.data):
                    data = isaac_ray_util.load_tensorboard_logs(self.tensorboard_logdir)
                    sleep(2)  # Lazy report metrics to avoid performance overhead

            self.data = data
            self.data["done"] = False
            return data

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        model_name = self.isaac_logdir.split("/")[-1]
        checkpoint = self.isaac_logdir + f"/{self.experiment_name}/nn/{model_name}.pth"
        subprocess.call(["cp", f"{checkpoint}", f"{checkpoint_dir}"])

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        model_name = checkpoint_dir.split("/")[-2]
        self.checkpoint = checkpoint_dir + f"/{model_name}.pth"

    def default_resource_request(self):
        resources = isaac_ray_util.get_gpu_node_resources(one_node_only=True)
        return tune.PlacementGroupFactory([{"CPU": resources["CPU"], "GPU": resources["GPU"]}], strategy="STRICT_PACK")


def invoke_tuning_run(cfg: dict, args: argparse.Namespace) -> None:
    """Invoke an Isaac-Ray tuning run optimized for Kubernetes deployment
    Args:
        cfg: Configuration dictionary extracted from job setup
        args: Command-line arguments related to tuning.
    """
    # Allow for early exit
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    # Get available resources
    resources = isaac_ray_util.get_gpu_node_resources()
    print(f"[INFO]: Available resources {resources}")

    if not ray.is_initialized():
        ray.init(
            address=args.ray_address,
            log_to_driver=True,
            num_gpus=len(resources),
        )

    print(f"[INFO]: Using config {cfg}")

    # Configure the search algorithm and the repeater
    searcher = OptunaSearch(
        metric=args.metric,
        mode=args.mode,
    )
    repeat_search = Repeater(searcher, repeat=args.repeat_run_count)

    if args.run_mode == "local":  # Standard config, to file
        run_config = air.RunConfig(
            name=f"IsaacRay-{args.cfg_class}-tune",
            storage_path=args.storage_path,
            verbose=1,
        )

    elif args.run_mode == "remote":  # MFlow, to MFlow server ;)
        run_config = ray.train.RunConfig(
            storage_path=args.mlflow_uri,
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=args.mlflow_uri,
                    experiment_name=f"IsaacRay-{args.cfg_class}-tune",
                    save_artifact=True,
                )
            ],
        )

    # Configure the tuning job
    tuner = tune.Tuner(
        IsaacLabTuneTrainable,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            search_alg=repeat_search,
            num_samples=args.num_samples,
            reuse_actors=True,
        ),
        run_config=run_config,
    )

    # Execute the tuning
    results = tuner.fit()

    # Save results to mounted volume
    if args.run_mode == "local":
        print("[DONE!]: Check results with tensorboard")
    else:
        print("[DONE!]: Check results with MLFlow ")


class JobCfg:
    """To be compatible with :meth: invoke_tuning_run and :class:IsaacLabTuneTrainable,
    at a minimum, the tune job should inherit from this class."""

    def __init__(self, cfg):
        assert "runner_args" in cfg, "No runner arguments specified."
        assert "--task" in cfg["runner_args"], "No task specified."
        assert "workflow" in cfg, "No workflow specified."
        assert "hydra_args" in cfg, "No hypeparameters specified."
        self.cfg = cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Cartpole.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="hyperparameter_tuning/vision_cartpole_cfg.py",
        required=False,
        help="The relative filepath where a hyperparameter sweep is defined",
    )
    parser.add_argument(
        "--cfg_class",
        type=str,
        default="CartpoleRGBNoTuneJobCfg",
        required=False,
        help="Name of the hyperparameter sweep class to use",
    )
    parser.add_argument(
        "--run_mode",
        choices=["local", "remote"],
        default="remote",
        help=(
            "Set to local to use ./isaaclab.sh -p python, set to "
            "remote to use /workspace/isaaclab/isaaclab.sh -p python"
        ),
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default=None,
        required=False,
        help=(
            "Where to store experiments. Can be directory for local dev, "
            "Must be a bucket your cluster has access to for remote."
        ),
    )

    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        required=False,
        help="The MLFlow Uri.",
    )

    parser.add_argument("--metric", type=str, default="rewards/time", help="What metric to tune for.")

    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="What to optimize the metric to while tuning",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="How many hyperparameter runs to try total.",
    )
    parser.add_argument(
        "--repeat_run_count",
        type=int,
        default=3,
        help="How many times to repeat each hyperparameter config.",
    )

    args = parser.parse_args()

    if args.run_mode == "remote":
        BASE_DIR = DOCKER_PREFIX  # ensure logs are dumped to persistent location
        PYTHON_EXEC = DOCKER_PREFIX + PYTHON_EXEC[2:]
        RL_GAMES_WORKFLOW = DOCKER_PREFIX + RL_GAMES_WORKFLOW
        print(f"[INFO]: Using remote mode {PYTHON_EXEC = } {RL_GAMES_WORKFLOW = }")

        if args.mlflow_uri is not None:
            import mlflow
            from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
        else:
            raise ValueError("Please provide a result MFLow URI server.")

    file_path = args.cfg_file
    class_name = args.cfg_class
    print(f"[INFO]: Attempting to use sweep config from {file_path = } {class_name = }")
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"[INFO]: Successfully imported {module_name} from {file_path}")
    if hasattr(module, class_name):
        ClassToInstantiate = getattr(module, class_name)
        print(f"[INFO]: Found correct class {ClassToInstantiate}")
        instance = ClassToInstantiate()
        print(f"[INFO]: Successfully instantiated class '{class_name}' from {file_path}")
        cfg = instance.cfg
        print(f"[INFO]: Grabbed the following hyperparameter sweep config: \n {cfg}")
        invoke_tuning_run(cfg, args)

    else:
        raise AttributeError(f"[ERROR]:Class '{class_name}' not found in {file_path}")
