# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import sys

# Allow for import of items from the ray workflow.
UTIL_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(UTIL_DIR))
import isaac_ray_tune
import isaac_ray_util
from ray import tune


class RLGamesCameraJobCfg(isaac_ray_tune.JobCfg):
    """In order to be compatible with :meth: invoke_tuning_run, and
    :class:IsaacLabTuneTrainable , configurations should
    be in a similar format to this class. This class can vary env count/horizon length,
    CNN structure, and MLP structure. Broad possible ranges are set, the specific values
    that work can be found via tuning. Tuning results can inform better ranges for a second tuning run."""

    def __init__(self, cfg={}, vary_env_count: bool = False, vary_cnn: bool = False, vary_mlp: bool = False):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        # Set up basic runner args
        cfg["runner_args"]["headless_singleton"] = tune.choice(["--headless"])
        cfg["runner_args"]["enable_cameras_singleton"] = tune.choice(["--enable_cameras"])
        cfg["workflow"] = tune.choice([isaac_ray_tune.RL_GAMES_WORKFLOW])
        cfg["hydra_args"]["agent.params.config.max_epochs"] = tune.choice([200])

        if vary_env_count:  # Vary the env count, and horizon length, and select a compatible mini-batch size
            # Check from 512 to 8196 envs in powers of 2
            # check horizon lengths of 8 to 256
            # More envs should be better, but different batch sizes can improve gradient estimation
            def batch_size_divisors(batch_size, min_size=128):
                return [i for i in range(1, batch_size + 1) if batch_size % i == 0 and i > min_size]

            cfg["runner_args"]["--num_envs"] = tune.choice([2**x for x in range(9, 13)])
            cfg["hydra_args"]["agent.params.config.horizon_length"] = tune.choice([2**x for x in range(3, 8)])
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
            if vary_cnn:  # Vary the depth, and size of the layers in the CNN part of the agent
                num_layers = tune.randint(2, 5)
                cfg["hydra_args"]["agent.params.network.cnn.type"] = tune.choice(["conv2d"])
                cfg["hydra_args"]["agent.params.network.cnn.activation"] = tune.choice(["relu", "elu"])
                cfg["hydra_args"]["agent.params.network.cnn.initializer"] = tune.choice(["{name:default}"])
                cfg["hydra_args"]["agent.params.network.cnn.regularizer"] = tune.choice(["{name:None}"])
                cfg["hydra_args"]["agent.params.network.cnn.convs"] = tune.sample_from(
                    lambda _: [
                        {
                            "filters": tune.randint(16, 32).sample(),
                            "kernel_size": tune.choice(["3", "4", "6", "8"]).sample(),
                            "strides": tune.choice(["1", "2", "3", "4"]).sample(),
                            "padding": tune.choice(["0", "1"]).sample(),
                        }
                        for _ in range(num_layers.sample())
                    ]
                )

        if vary_mlp:
            # Vary the MLP structure; neurons (units) per layer, number of layers,
            def generate_mlp_layer():
                return {
                    "units": tune.randint(2**3, 2**12),
                    "activation": tune.choice(["relu", "tanh", "sigmoid", "elu"]),
                    "initializer": {"name": tune.choice(["default", "he_uniform", "glorot_uniform"])},
                }

            cfg["hydra_args"]["agent.params.network.mlp"] = {
                "layers": tune.sample_from(
                    lambda _: [generate_mlp_layer() for _ in range(tune.randint(1, 10).sample())]
                )
            }
        super().__init__(cfg)


class RLGamesResNetCameraJob(RLGamesCameraJobCfg):
    """Try different ResNet sizes."""

    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["hydra_args"]["env.observations.policy.image.params.model_name"] = tune.choice(
            ["resnet18", "resnet34", "resnet50", "resnet101"]
        )
        super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)


class RLGamesTheiaCameraJob(RLGamesCameraJobCfg):
    """Try diferent Theia sizes."""

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
