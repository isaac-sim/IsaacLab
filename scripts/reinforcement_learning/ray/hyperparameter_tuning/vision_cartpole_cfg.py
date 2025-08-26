# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys
from typing import Any

# Allow for import of items from the ray workflow.
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])
import util
import vision_cfg
from ray import tune
from ray.tune.stopper import Stopper


class CartpoleRGBNoTuneJobCfg(vision_cfg.CameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=False, vary_mlp=False)


class CartpoleRGBCNNOnlyJobCfg(vision_cfg.CameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=True, vary_mlp=False)


class CartpoleRGBJobCfg(vision_cfg.CameraJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-v0"])
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


class CartpoleResNetJobCfg(vision_cfg.ResNetCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-ResNet18-v0"])
        super().__init__(cfg)


class CartpoleTheiaJobCfg(vision_cfg.TheiaCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-TheiaTiny-v0"])
        super().__init__(cfg)


class CartpoleEarlyStopper(Stopper):
    def __init__(self):
        self._bad_trials = set()

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        iter = result.get("training_iteration", 0)
        out_of_bounds = result.get("Episode/Episode_Termination/cart_out_of_bounds")

        # Mark the trial for stopping if conditions are met
        if 20 <= iter and out_of_bounds is not None and out_of_bounds > 0.85:
            self._bad_trials.add(trial_id)

        return trial_id in self._bad_trials

    def stop_all(self) -> bool:
        return False  # only stop individual trials
