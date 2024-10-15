# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import time

import isaac_ray_util

# import ray
from ray import tune


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
            return data

    def save_checkpoint(self, checkpoint_dir):
        # Not sure if this is even needed tbh
        model_name = self.logdir.split("/")[-2]
        checkpoint = self.logdir / "nn" / model_name + ".pth"
        subprocess.call(["cp", f"{checkpoint}", f"{checkpoint_dir}"])

    def load_checkpoint(self, checkpoint_dir):
        model_name = checkpoint_dir.split("/")[-2]
        self.checkpoint = checkpoint_dir / model_name + ".pth"


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
