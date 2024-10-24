# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import pathlib
import sys

UTIL_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(UTIL_DIR))

import isaac_ray_tune


class CartpoleRGBNoTuneJobCfg(isaac_ray_tune.JobCfg):  # Idempotent
    def __init__(self, cfg: dict = {}):
        cfg["workflow"] = "/workspace/isaaclab/workflows/rl_games/train.py"
        cfg["runner_args"]["singletons"] = ["--headless", "--enable_cameras"]
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-RGB-Camera-Direct-v0"
        cfg["hydra_args"] = {}
        super().__init__(cfg, vary_env_count=False, vary_cnn=False, vary_mlp=False)


class CartpoleRGBJobCfg(isaac_ray_tune.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-RGB-Camera-v0"
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


class CartpoleResNetJobCfg(isaac_ray_tune.RLGamesResNetCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-ResNet18-Camera-v0"
        super().__init__(cfg)


class CartpoleTheiaJobCfg(isaac_ray_tune.RLGamesTheiaCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-TheiaTiny-Camera-v0"
        super().__init__(cfg)
