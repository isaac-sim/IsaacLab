# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys

# Allow for import of items from the ray workflow.
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])

import isaac_ray_util
import rl_games_vision_cfg
from ray import tune


class CartpoleRGBNoTuneJobCfg(rl_games_vision_cfg.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=False, vary_mlp=False)


class CartpoleRGBCNNOnly(rl_games_vision_cfg.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=True, vary_mlp=False)


class CartpoleRGBJobCfg(rl_games_vision_cfg.RLGamesCameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-v0"])
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


class CartpoleResNetJobCfg(rl_games_vision_cfg.RLGamesResNetCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-ResNet18-v0"])
        super().__init__(cfg)


class CartpoleTheiaJobCfg(rl_games_vision_cfg.RLGamesTheiaCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = isaac_ray_util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-TheiaTiny-v0"])
        super().__init__(cfg)
