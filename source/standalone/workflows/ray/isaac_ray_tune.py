# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import os
import subprocess
import sys
import time

import isaac_ray_util
import ray
from ray import air, tune

# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

# from ray.tune.search.concurrency_limiter import ConcurrencyLimiter
from ray.tune.search.repeater import Repeater


class IsaacLabTuneTrainable(tune.Trainable):
    def setup(self, config):
        self.data = None

        if hasattr(self, "checkpoint"):
            config["runner_args"]["checkpoint_singleton"] = self.checkpoint
        experiment = isaac_ray_util.invoke_run(config)
        self.proc = experiment["proc"]
        self.experiment_name = experiment["experiment_name"]
        self.logdir = experiment["logdir"]
        self.tensorboard_logdir = self.logdir / "summaries"

    def step(self):
        if self.proc is None:  # failed to start, return negative signal
            raise RuntimeError("Could not start desired trial.")
        if self.proc.poll() is not None:
            return {"Done": True}
        else:
            data = isaac_ray_util.load_tensorboard_logs(self.tensorboard_logdir)
            while self.data == data:
                time.sleep(0.1)  # avoid busy wait...
                data = isaac_ray_util.load_tensorboard_logs(self.tensorboard_logdir)
            print(f"New data {data}")
            return data

    def save_checkpoint(self, checkpoint_dir):
        # Not sure if this is even needed tbh
        model_name = self.logdir.split("/")[-2]
        checkpoint = self.logdir / "nn" / model_name + ".pth"
        subprocess.call(["cp", f"{checkpoint}", f"{checkpoint_dir}"])

    def load_checkpoint(self, checkpoint_dir):
        model_name = checkpoint_dir.split("/")[-2]
        self.checkpoint = checkpoint_dir / model_name + ".pth"


def invoke_tuning_run(
    cfg,
    metric="rewards/time",
    mode="max",
):
    ray.init(address="auto")  # Initialize Ray
    resources = isaac_ray_util.get_total_gpu_node_resources()
    print(f"[INFO]: Resources per worker: {resources}")
    # Define trainable with specific resource allocation
    isaac_lab_trainable_with_resources = tune.with_resources(
        IsaacLabTuneTrainable,  # Make sure IsaacLabTuneTrainable is defined and imported
        resources,
    )

    # Define BOHB Search Algorithm
    bohb_search = TuneBOHB(
        metric=metric,
        mode=mode,
    )

    # Repeat each configuration 3 times
    repeat_search = Repeater(bohb_search, repeat=3)

    # Running the experiment using the new Ray Tune API
    tuner = tune.Tuner(
        isaac_lab_trainable_with_resources,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            search_alg=repeat_search,
            scheduler=None,  # No scheduler is used
            num_samples=args.num_samples,  # Ensure args.num_samples is well-defined
        ),
        run_config=air.RunConfig(
            name="BOHB_test", local_dir="./ray_results", verbose=1, failure_config=air.FailureConfig(fail_fast=True)
        ),
    )

    # Execute the tuning
    results = tuner.fit()

    # Output the best hyperparameters
    print("Best hyperparameters found were: ", results.get.best_result().config)


# You can add the args parsing and cfg loading part here if necessary.


class JobCfg:
    def __init__(self, cfg):
        assert "runner_args" in cfg, "No runner arguments specified."
        assert "workflow" in cfg, "No workflow specified."
        assert "hydra_args" in cfg, "No hypeparameters specified."
        self.cfg = cfg


class RLGamesCameraJobCfg(JobCfg):
    def __init__(self, cfg={}, vary_env_count: bool = False, vary_cnn: bool = False, vary_mlp: bool = False):
        # Set up basic runner args
        cfg["runner_args"]["singletons"] = ["--headless", "--enable_cameras"]
        cfg["workflow"] = "/workspace/isaaclab/workflows/rl_games/train.py"

        cfg["hydra_args"]["agent.params.config.save_best_after"] = 5
        cfg["hydra_args"]["agent.params.config.save_frequency"] = 5

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

            if vary_mlp:
                cfg["hydra_args"]["agents.params.network.mlp"] = {
                    "layers": tune.sample_from(
                        lambda _: [generate_mlp_layer() for _ in range(tune.randint(1, 10).sample())]
                    )
                }
        super().__init__(cfg)


class RLGamesResNetCameraJob(RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg["hydra_args"]["env.observations.policy.image.params.model_name"] = tune.choice(
            ["resnet18", "resnet34", "resnet50", "resnet101"]
        )
        super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)


class RLGamesTheiaCameraJob(RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
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
    isaac_ray_util.add_cluster_args(parser=parser)

    parser.add_argument(
        "--train_task_no_tune",
        type=str,
        required=False,
        help="Train a single RL games task without tuning for testing purposes",
    )

    parser.add_argument(
        "--hyperparam_cfg_file",
        type=str,
        required=False,
        help="The relative filepath where a hyperparameter sweep is defined",
    )
    parser.add_argument(
        "--hyperparam_cfg_class", type=str, required=False, help="Name of the hyperparameter sweep class to use"
    )
    args = parser.parse_args()

    if args.train_task is not None:
        print(f"Training one task only: {args.train_task}")
        cfg_cls = RLGamesCameraJobCfg(vary_env_count=False, vary_cnn=False, vary_mlp=False)
        cfg_cls.cfg["runner_args"]["--task"] = args.train_task

    if args.hyperparam_cfg_file is not None and args.train_task is None:
        file_path = args.hyperparam_cfg_file
        class_name = args.hyperparam_cfg_class
        print(f"Attempting to use sweep config from {file_path = } {class_name = }")

        # Extract the module name from the file name
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # Step 2: Dynamically load the module from the file
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            print(f"Successfully imported {module_name} from {file_path}")

            # Step 3: Check and instantiate the class from the module
            if hasattr(module, class_name):
                ClassToInstantiate = getattr(module, class_name)  # Get the class by name
                instance = ClassToInstantiate()  # Instantiate the class
                print(f"Successfully instantiated class '{class_name}' from {file_path}")
                # You can now interact with the instance, for example:
                # instance.some_method()
            else:
                print(f"Class '{class_name}' not found in {file_path}")

        except FileNotFoundError:
            print(f"File {file_path} not found")
        except Exception as e:
            print(f"Failed to import module or instantiate class from {file_path}: {e}")
    else:
        print("No file provided for importing.")
