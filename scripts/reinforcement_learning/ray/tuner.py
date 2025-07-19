# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import importlib.util
import os
import subprocess
import sys
from time import sleep, time

import ray
import util
from ray import air, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.repeater import Repeater

"""
This script breaks down an aggregate tuning job, as defined by a hyperparameter sweep configuration,
into individual jobs (shell commands) to run on the GPU-enabled nodes of the cluster.
By default, one worker is created for each GPU-enabled node in the cluster for each individual job.
To use more than one worker per node (likely the case for multi-GPU machines), supply the
num_workers_per_node argument.

Each hyperparameter sweep configuration should include the workflow,
runner arguments, and hydra arguments to vary.

This assumes that all workers in a cluster are homogeneous. For heterogeneous workloads,
create several heterogeneous clusters (with homogeneous nodes in each cluster),
then submit several overall-cluster jobs with :file:`../submit_job.py`.
KubeRay clusters on Google GKE can be created with :file:`../launch.py`

To report tune metrics on clusters, a running MLFlow server with a known URI that the cluster has
access to is required. For KubeRay clusters configured with :file:`../launch.py`, this is included
automatically, and can be easily found with with :file:`grok_cluster_with_kubectl.py`

Usage:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/ray/tuner.py -h

    # Examples
    # Local
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/tuner.py --run_mode local \
    --cfg_file scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py \
    --cfg_class CartpoleTheiaJobCfg
    # Remote (run grok cluster or create config file mentioned in :file:`submit_job.py`)
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/submit_job.py \
    --aggregate_jobs tuner.py \
    --cfg_file hyperparameter_tuning/vision_cartpole_cfg.py \
    --cfg_class CartpoleTheiaJobCfg --mlflow_uri <MLFLOW_URI_FROM_GROK_OR_MANUAL>

"""

DOCKER_PREFIX = "/workspace/isaaclab/"
BASE_DIR = os.path.expanduser("~")
PYTHON_EXEC = "./isaaclab.sh -p"
WORKFLOW = "scripts/reinforcement_learning/rl_games/train.py"
NUM_WORKERS_PER_NODE = 1  # needed for local parallelism
PROCESS_RESPONSE_TIMEOUT = 200.0  # seconds to wait before killing the process when it stops responding
MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS = 1000  # maximum number of lines to read from the training process logs
MAX_LOG_EXTRACTION_ERRORS = 2  # maximum allowed LogExtractionErrors before we abort the whole training


class IsaacLabTuneTrainable(tune.Trainable):
    """The Isaac Lab Ray Tune Trainable.
    This class uses the standalone workflows to start jobs, along with the hydra integration.
    This class achieves Ray-based logging through reading the tensorboard logs from
    the standalone workflows. This depends on a config generated in the format of
    :class:`JobCfg`
    """

    def setup(self, config: dict) -> None:
        """Get the invocation command, return quick for easy scheduling."""
        self.data = None
        self.time_since_last_proc_response = 0.0
        self.invoke_cmd = util.get_invocation_command_from_cfg(cfg=config, python_cmd=PYTHON_EXEC, workflow=WORKFLOW)
        print(f"[INFO]: Recovered invocation with {self.invoke_cmd}")
        self.experiment = None

    def reset_config(self, new_config: dict):
        """Allow environments to be re-used by fetching a new invocation command"""
        self.setup(new_config)
        return True

    def step(self) -> dict:
        if self.experiment is None:  # start experiment
            # When including this as first step instead of setup, experiments get scheduled faster
            # Don't want to block the scheduler while the experiment spins up
            print(f"[INFO]: Invoking experiment as first step with {self.invoke_cmd}...")
            try:
                experiment = util.execute_job(
                    self.invoke_cmd,
                    identifier_string="",
                    extract_experiment=True,  # Keep this as True to return a valid dictionary
                    persistent_dir=BASE_DIR,
                    max_lines_to_search_logs=MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS,
                    max_time_to_search_logs=PROCESS_RESPONSE_TIMEOUT,
                )
            except util.LogExtractionError:
                self.data = {
                    "LOG_EXTRACTION_ERROR_STOPPER_FLAG": True,
                    "done": True,
                }
                return self.data
            self.experiment = experiment
            print(f"[INFO]: Tuner recovered experiment info {experiment}")
            self.proc = experiment["proc"]
            self.experiment_name = experiment["experiment_name"]
            self.isaac_logdir = experiment["logdir"]
            self.tensorboard_logdir = self.isaac_logdir + "/" + self.experiment_name
            self.done = False

        if self.proc is None:
            raise ValueError("Could not start trial.")
        proc_status = self.proc.poll()
        if proc_status is not None:  # process finished, signal finish
            self.data["done"] = True
            print(f"[INFO]: Process finished with {proc_status}, returning...")
        else:  # wait until the logs are ready or fresh
            data = util.load_tensorboard_logs(self.tensorboard_logdir)

            while data is None:
                data = util.load_tensorboard_logs(self.tensorboard_logdir)
                proc_status = self.proc.poll()
                if proc_status is not None:
                    break
                sleep(2)  # Lazy report metrics to avoid performance overhead

            if self.data is not None:
                data_ = {k: v for k, v in data.items() if k != "done"}
                self_data_ = {k: v for k, v in self.data.items() if k != "done"}
                unresponsiveness_start_time = time()
                while util._dicts_equal(data_, self_data_):
                    self.time_since_last_proc_response = time() - unresponsiveness_start_time
                    data = util.load_tensorboard_logs(self.tensorboard_logdir)
                    data_ = {k: v for k, v in data.items() if k != "done"}
                    proc_status = self.proc.poll()
                    if proc_status is not None:
                        break
                    if self.time_since_last_proc_response > PROCESS_RESPONSE_TIMEOUT:
                        self.time_since_last_proc_response = 0.0
                        print("[WARNING]: Training workflow process is not responding, terminating...")
                        self.proc.terminate()
                        try:
                            self.proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            print("[ERROR]: The process did not terminate within timeout duration.")
                            self.proc.kill()
                            self.proc.wait()
                        self.data = data
                        self.data["done"] = True
                        return self.data
                    sleep(2)  # Lazy report metrics to avoid performance overhead

            self.data = data
            self.data["done"] = False
        return self.data

    def default_resource_request(self):
        """How many resources each trainable uses. Assumes homogeneous resources across gpu nodes,
        and that each trainable is meant for one node, where it uses all available resources."""
        resources = util.get_gpu_node_resources(one_node_only=True)
        if NUM_WORKERS_PER_NODE != 1:
            print("[WARNING]: Splitting node into more than one worker")
        return tune.PlacementGroupFactory(
            [{"CPU": resources["CPU"] / NUM_WORKERS_PER_NODE, "GPU": resources["GPU"] / NUM_WORKERS_PER_NODE}],
            strategy="STRICT_PACK",
        )


class LogExtractionErrorStopper(tune.Stopper):
    """Stopper that stops all trials if multiple LogExtractionErrors occur.

    Args:
        max_errors: The maximum number of LogExtractionErrors allowed before terminating the experiment.
    """

    def __init__(self, max_errors: int):
        self.max_errors = max_errors
        self.error_count = 0

    def __call__(self, trial_id, result):
        """Increments the error count if trial has encountered a LogExtractionError.

        It does not stop the trial based on the metrics, always returning False.
        """
        if result.get("LOG_EXTRACTION_ERROR_STOPPER_FLAG", False):
            self.error_count += 1
            print(
                f"[ERROR]: Encountered LogExtractionError {self.error_count} times. "
                f"Maximum allowed is {self.max_errors}."
            )
        return False

    def stop_all(self):
        """Returns true if number of LogExtractionErrors exceeds the maximum allowed, terminating the experiment."""
        if self.error_count > self.max_errors:
            print("[FATAL]: Encountered LogExtractionError more than allowed, aborting entire tuning run... ")
            return True
        else:
            return False


def invoke_tuning_run(cfg: dict, args: argparse.Namespace) -> None:
    """Invoke an Isaac-Ray tuning run.

    Log either to a local directory or to MLFlow.
    Args:
        cfg: Configuration dictionary extracted from job setup
        args: Command-line arguments related to tuning.
    """
    # Allow for early exit
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    print("[WARNING]: Not saving checkpoints, just running experiment...")
    print("[INFO]: Model parameters and metrics will be preserved.")
    print("[WARNING]: For homogeneous cluster resources only...")
    # Get available resources
    resources = util.get_gpu_node_resources()
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
            storage_path="/tmp/ray",
            name=f"IsaacRay-{args.cfg_class}-tune",
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=0,  # Disable periodic checkpointing
                checkpoint_at_end=False,  # Disable final checkpoint
            ),
            stop=LogExtractionErrorStopper(max_errors=MAX_LOG_EXTRACTION_ERRORS),
        )

    elif args.run_mode == "remote":  # MLFlow, to MLFlow server
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=args.mlflow_uri,
            experiment_name=f"IsaacRay-{args.cfg_class}-tune",
            save_artifact=False,
            tags={"run_mode": "remote", "cfg_class": args.cfg_class},
        )

        run_config = ray.train.RunConfig(
            name="mlflow",
            storage_path="/tmp/ray",
            callbacks=[mlflow_callback],
            checkpoint_config=ray.train.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False),
            stop=LogExtractionErrorStopper(max_errors=MAX_LOG_EXTRACTION_ERRORS),
        )
    else:
        raise ValueError("Unrecognized run mode.")

    # Configure the tuning job
    tuner = tune.Tuner(
        IsaacLabTuneTrainable,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            metric=args.metric,
            mode=args.mode,
            search_alg=repeat_search,
            num_samples=args.num_samples,
            reuse_actors=True,
        ),
        run_config=run_config,
    )

    # Execute the tuning
    tuner.fit()

    # Save results to mounted volume
    if args.run_mode == "local":
        print("[DONE!]: Check results with tensorboard dashboard")
    else:
        print("[DONE!]: Check results with MLFlow dashboard")


class JobCfg:
    """To be compatible with :meth: invoke_tuning_run and :class:IsaacLabTuneTrainable,
    at a minimum, the tune job should inherit from this class."""

    def __init__(self, cfg: dict):
        """
        Runner args include command line arguments passed to the task.
        For example:
        cfg["runner_args"]["headless_singleton"] = "--headless"
        cfg["runner_args"]["enable_cameras_singleton"] = "--enable_cameras"
        """
        assert "runner_args" in cfg, "No runner arguments specified."
        """
        Task is the desired task to train on. For example:
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-TheiaTiny-v0"])
        """
        assert "--task" in cfg["runner_args"], "No task specified."
        """
        Hydra args define the hyperparameters varied within the sweep. For example:
        cfg["hydra_args"]["agent.params.network.cnn.activation"] = tune.choice(["relu", "elu"])
        """
        assert "hydra_args" in cfg, "No hyperparameters specified."
        self.cfg = cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Isaac Lab hyperparameters.")
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
        "--workflow",
        default=None,  # populated with RL Games
        help="The absolute path of the workflow to use for the experiment. By default, RL Games is used.",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        required=False,
        help="The MLFlow Uri.",
    )
    parser.add_argument(
        "--num_workers_per_node",
        type=int,
        default=1,
        help="Number of workers to run on each GPU node. Only supply for parallelism on multi-gpu nodes",
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
    parser.add_argument(
        "--process_response_timeout",
        type=float,
        default=PROCESS_RESPONSE_TIMEOUT,
        help="Training workflow process response timeout.",
    )
    parser.add_argument(
        "--max_lines_to_search_experiment_logs",
        type=float,
        default=MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS,
        help="Max number of lines to search for experiment logs before terminating the training workflow process.",
    )
    parser.add_argument(
        "--max_log_extraction_errors",
        type=float,
        default=MAX_LOG_EXTRACTION_ERRORS,
        help="Max number number of LogExtractionError failures before we abort the whole tuning run.",
    )

    args = parser.parse_args()
    PROCESS_RESPONSE_TIMEOUT = args.process_response_timeout
    MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS = int(args.max_lines_to_search_experiment_logs)
    print(
        "[INFO]: The max number of lines to search for experiment logs before (early) terminating the training "
        f"workflow process is set to {MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS}.\n"
        "[INFO]: The process response timeout, used while updating tensorboard scalars and searching for "
        f"experiment logs, is set to {PROCESS_RESPONSE_TIMEOUT} seconds."
    )
    MAX_LOG_EXTRACTION_ERRORS = int(args.max_log_extraction_errors)
    print(
        "[INFO]: Max number of LogExtractionError failures before we abort the whole tuning run is "
        f"set to {MAX_LOG_EXTRACTION_ERRORS}.\n"
    )
    NUM_WORKERS_PER_NODE = args.num_workers_per_node
    print(f"[INFO]: Using {NUM_WORKERS_PER_NODE} workers per node.")
    if args.run_mode == "remote":
        BASE_DIR = DOCKER_PREFIX  # ensure logs are dumped to persistent location
        PYTHON_EXEC = DOCKER_PREFIX + PYTHON_EXEC[2:]
        if args.workflow is None:
            WORKFLOW = DOCKER_PREFIX + WORKFLOW
        else:
            WORKFLOW = args.workflow
        print(f"[INFO]: Using remote mode {PYTHON_EXEC=} {WORKFLOW=}")

        if args.mlflow_uri is not None:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            from ray.air.integrations.mlflow import MLflowLoggerCallback
        else:
            raise ValueError("Please provide a result MLFLow URI server.")
    else:  # local
        PYTHON_EXEC = os.getcwd() + "/" + PYTHON_EXEC[2:]
        if args.workflow is None:
            WORKFLOW = os.getcwd() + "/" + WORKFLOW
        else:
            WORKFLOW = args.workflow
        BASE_DIR = os.getcwd()
        print(f"[INFO]: Using local mode {PYTHON_EXEC=} {WORKFLOW=}")
    file_path = args.cfg_file
    class_name = args.cfg_class
    print(f"[INFO]: Attempting to use sweep config from {file_path=} {class_name=}")
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
