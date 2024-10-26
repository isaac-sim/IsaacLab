# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import importlib.util
import os
import subprocess
import sys

import isaac_ray_util
import ray
from ray import air, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.repeater import Repeater

"""
This script breaks down an aggregate tuning job, as defined by a hyperparameter sweep configuration,
into individual jobs (shell commands) to run on the GPU-enabled nodes of the cluster.
By default, (unless combined as a sub-job in a resource-wrapped aggregate job
or ``--num_workers_per_node`` is specified), one worker is created
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
        invoke_cmd = isaac_ray_util.get_invocation_command_from_cfg(cfg=config, python_cmd=PYTHON_EXEC)
        experiment = isaac_ray_util.execute_job(
            invoke_cmd, identifier_string="", extract_experiment=True, persistent_dir=BASE_DIR
        )
        print(f"[INFO]: Tuner recovered experiment info {experiment}")
        self.proc = experiment["proc"]
        self.experiment_name = experiment["experiment_name"]
        self.isaac_logdir = experiment["logdir"]
        self.tensorboard_logdir = self.isaac_logdir + f"/{self.experiment_name}/summaries"

    def step(self) -> dict:
        if self.proc is None:  # failed to start, return negative signal
            raise RuntimeError("Could not start desired trial.")
        if self.proc.poll() is not None:
            self.data["Done"] = True
            print("[INFO]: Process finished, returning...")
            return self.data
        else:
            data = isaac_ray_util.load_tensorboard_logs(self.tensorboard_logdir)
            print(f"{data = }")
            self.data = data
            return data

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        model_name = self.isaac_logdir.split("/")[-1]
        checkpoint = self.isaac_logdir + f"/{self.experiment_name}/nn/{model_name}.pth"
        subprocess.call(["cp", f"{checkpoint}", f"{checkpoint_dir}"])

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        model_name = checkpoint_dir.split("/")[-2]
        self.checkpoint = checkpoint_dir + f"/{model_name}.pth"


def invoke_tuning_run(
    cfg: dict,
    storage_path: str,
    num_workers_per_node: int = 1,
    metric: str = "rewards/time",
    mode: str = "max",
    num_samples: int = 1000,
    repeat_run_count: int = 3,
    ray_address: str = "auto",
) -> None:
    """Invoke an Isaac-Ray tuning run

    Args:
        cfg: A configuration extracted from :class:JobCfg, similar in format to :class:RLGamesCameraJobCfg
        storage_path: Either a local path on a single-node tuning workflow, or a bucket that
            a cluster has access to for a multi-node workflow.
        num_workers_per_node: how many workers to split each node into
        metric: The metric to tune to optimize. Defaults to "rewards/time".
        mode: Which mode to run the optimization. Defaults to "max".
        num_samples: The number of total samples to draw from
            the search space when tuning. Equal to the total number of runs. Defaults to 1000.
        repeat_run_count: The number of times to repeat each hyperparameter
            configuration. Defaults to 3.
        ray_address: What ray address to connect to. Defaults to 'auto'
    """
    if not ray.is_initialized():
        ray.init(address=ray_address, log_to_driver=True)
    resources = isaac_ray_util.get_gpu_node_resources(one_node_only=True)

    for key, value in resources:
        resources[key] = value / num_workers_per_node

    print(f"[INFO]: Resources per worker: {resources}")
    print(f"[INFO]: Using config {cfg}")
    # Define trainable with specific resource allocation
    isaac_lab_trainable_with_resources = tune.with_resources(
        IsaacLabTuneTrainable,  # Make sure IsaacLabTuneTrainable is defined and imported
        resources,
    )

    # Search Algorithm # TODO: Support other search algorithms, and Scheduling option instead of repeater
    searcher = OptunaSearch(
        metric=metric,
        mode=mode,  # ,
    )

    # Repeat each configuration 3 times
    repeat_search = Repeater(searcher, repeat=repeat_run_count)

    # Running the experiment using the new Ray Tune API
    tuner = tune.Tuner(
        isaac_lab_trainable_with_resources,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            search_alg=repeat_search,
            scheduler=None,  # No scheduler is used to be compatible with Repeater.
            num_samples=num_samples,  # Ensure args.num_samples is well-defined
        ),
        run_config=air.RunConfig(
            name="Isaac-Ray;-)", storage_path=storage_path, verbose=1, failure_config=air.FailureConfig(fail_fast=True)
        ),
    )

    # Execute the tuning
    results = tuner.fit()

    # Output the best hyperparameters
    print(f"Best hyperparameters found were: {results.get_best_result(mode=mode, metric=metric)}")
    print(results.get_dataframe())


class JobCfg:
    """To be compatible with :meth: invoke_tuning_run and :class:IsaacLabTuneTrainable,
    at a minimum, the tune job should inherit from this class."""

    def __init__(self, cfg):
        assert "runner_args" in cfg, "No runner arguments specified."
        assert "workflow" in cfg, "No workflow specified."
        assert "hydra_args" in cfg, "No hypeparameters specified."
        self.cfg = cfg


class RLGamesCameraJobCfg(JobCfg):
    """In order to be compatible with :meth: invoke_tuning_run, and
    :class:IsaacLabTuneTrainable , configurations should
    be in a similar format to this class."""

    def __init__(self, cfg={}, vary_env_count: bool = False, vary_cnn: bool = False, vary_mlp: bool = False):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        # Set up basic runner args
        cfg["runner_args"]["headless_singleton"] = tune.choice(["--headless"])
        cfg["runner_args"]["enable_cameras_singleton"] = tune.choice(["--enable_cameras"])
        cfg["workflow"] = tune.choice([RL_GAMES_WORKFLOW])

        cfg["hydra_args"]["agent.params.config.save_best_after"] = tune.choice([5])
        cfg["hydra_args"]["agent.params.config.save_frequency"] = tune.choice([5])

        if vary_env_count:

            def batch_size_divisors(batch_size, min_size=128):
                return [i for i in range(1, batch_size + 1) if batch_size % i == 0 and i > min_size]

            cfg["runner_args"]["--num_envs"] = tune.randint(2**6, 2**14 + 1)
            cfg["hydra_args"]["agent.params.config.horizon_length"] = tune.randint(1, 200)
            cfg["hydra_args"]["agent.params.config.minibatch_size.config"] = (
                tune.sample_from(
                    lambda spec: tune.choice(
                        batch_size_divisors(
                            spec.config.hydra_args["agent.params.config.horizon_length"]
                            * spec.config.runner_args["--num_envs"]
                            * spec.config.runner_args["--num_envs"]
                        )
                    )
                ),
            )
        if vary_cnn:

            def generate_cnn_layer():
                return {
                    "filters": tune.randint(2**4, 2**9 + 1),
                    "kernel_size": tune.randint(2**1, 2**4 + 1),
                    "strides": tune.randint(2**1, 2**4 + 1),
                    "padding": tune.choice([0, 1]),  # Padding remains as a discrete choice
                }

            cfg["hydra_args"]["agents.params.network.cnn"] = {
                "type": "conv2d",
                "activation": tune.choice(["relu", "tanh", "sigmoid"]),
                "initializer": {"name": tune.choice(["default", "he_uniform", "glorot_uniform"])},
                "regularizer": {
                    "name": tune.choice([None, "l2", "l1"]),
                },
                "convs": [generate_cnn_layer() for _ in range(tune.randint(1, 6).sample())],
            }

        if vary_mlp:

            def generate_mlp_layer():
                return {
                    "units": tune.randint(2**3, 2**12),
                    "activation": tune.choice(["relu", "tanh", "sigmoid", "elu"]),
                    "initializer": {"name": tune.choice(["default", "he_uniform", "glorot_uniform"])},
                }

            cfg["hydra_args"]["agents.params.network.mlp"] = {
                "layers": tune.sample_from(
                    lambda _: [generate_mlp_layer() for _ in range(tune.randint(1, 10).sample())]
                )
            }
        super().__init__(cfg)


class RLGamesResNetCameraJob(RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["hydra_args"]["env.observations.policy.image.params.model_name"] = tune.choice(
            ["resnet18", "resnet34", "resnet50", "resnet101"]
        )
        super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)


class RLGamesTheiaCameraJob(RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["hydra_args"]["env.observations.policy.image.params.model_name"] = tune.choice([
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ])
        super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Cartpole.")
    parser.add_argument("--tune_type", choices=["standard_no_tune", "standard", "resnet", "theia"])
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
        "--mode",
        choices=["local", "docker"],
        default="docker",
        help=(
            "Set to local to use ./isaaclab.sh -p python, set to "
            "remote to use /workspace/isaaclab/isaaclab.sh -p python"
        ),
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default=os.path.expanduser("~/isaac_ray_logs"),
        required=False,
        help=(
            "Where to store experiments. Can be directory for local dev, "
            "Must be a bucket your cluster has access to for remote."
        ),
    )
    parser.add_argument(
        "--num_workers_per_node",
        type=int,
        help="Supply to split each node into num_workers evenly.",
    )

    args = parser.parse_args()

    if args.mode == "docker":
        BASE_DIR = DOCKER_PREFIX  # ensure logs are dumped to persistent location
        PYTHON_EXEC = DOCKER_PREFIX + PYTHON_EXEC[2:]
        RL_GAMES_WORKFLOW = DOCKER_PREFIX + RL_GAMES_WORKFLOW
        print(f"[INFO]: Using docker mode {PYTHON_EXEC = } {RL_GAMES_WORKFLOW = }")

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
        invoke_tuning_run(
            cfg,
            storage_path=args.storage_path,
            num_workers_per_node=args.num_workers_per_node,
            ray_address=args.ray_address,
        )

    else:
        raise AttributeError(f"[ERROR]:Class '{class_name}' not found in {file_path}")
