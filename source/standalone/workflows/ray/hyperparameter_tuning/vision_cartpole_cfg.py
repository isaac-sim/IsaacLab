# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys

UTIL_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(UTIL_DIR))
import isaac_ray_tune
import isaac_ray_util
from ray import tune


class CartpoleRGBNoTuneJobCfg(isaac_ray_tune.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=False, vary_mlp=False)


class CartpoleRGBJobCfg(isaac_ray_tune.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-Camera-v0"])
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


class CartpoleResNetJobCfg(isaac_ray_tune.RLGamesResNetCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-ResNet18-Camera-v0"])
        super().__init__(cfg)


class CartpoleTheiaJobCfg(isaac_ray_tune.RLGamesTheiaCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-TheiaTiny-Camera-v0"])
        super().__init__(cfg)
