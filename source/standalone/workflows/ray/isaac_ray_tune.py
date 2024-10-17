# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import time

import isaac_ray_util
import ray
from ray import air, tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter
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
            return {"Done": True, "reward": -1}
        elif self.proc.poll() is not None:
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


def invoke_tuning_run(args, cfg):
    num_gpu_per_worker = args.cluster_gpu_count // args.num_workers
    num_cpu_per_worker = args.cluster_cpu_count // args.num_workers
    ray.init(address="auto")

    # Define trainable with specific resource allocation
    isaac_lab_trainable_with_resources = tune.with_resources(
        IsaacLabTuneTrainable, {"cpu": num_cpu_per_worker, "gpu": num_gpu_per_worker}
    )

    # Define BOHB Search Algorithm
    bohb_search = TuneBOHB(
        metric="rewards/time",
        mode="max",
    )

    # Define HyperBand Scheduler tailored for BOHB
    hyperband_scheduler = HyperBandForBOHB(
        time_attr="info/epochs", max_t=args.max_iterations, reduction_factor=2, stop_last_trials=False
    )

    # Limit the number of concurrent trials
    limited_search = ConcurrencyLimiter(bohb_search, max_concurrent=4)

    # Repeat each configuration 3 times
    repeat_search = Repeater(limited_search, repeat=3)

    # Running the experiment using the new Ray Tune API
    tuner = tune.Tuner(
        isaac_lab_trainable_with_resources,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            search_alg=repeat_search,
            scheduler=hyperband_scheduler,
            num_samples=args.num_samples,
        ),
        run_config=air.RunConfig(
            name="BOHB_test", local_dir="./ray_results", verbose=1, failure_config=air.FailureConfig(fail_fast=True)
        ),
    )

    # Get results
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


# You can add the args parsing and cfg loading part here if necessary.


class JobCfg:
    def __init__(self, cfg):
        assert "runner_args" in cfg, "No runner arguments specified."
        assert "workflow" in cfg, "No workflow specified."
        assert "hydra_args" in cfg, "No hypeparameters specified."
        self.cfg = cfg


class RLGamesCameraJobCfg(JobCfg):
    def __init__(self, cfg={}, vary_env_count: bool = True, vary_cnn: bool = False, vary_mlp: bool = True):
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
    pass
