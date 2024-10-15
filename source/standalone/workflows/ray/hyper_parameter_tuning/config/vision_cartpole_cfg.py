# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from ray import tune
import sys

UTIL_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(UTIL_DIR))

import isaac_ray_util

class JobCfg():
    def __init__(self, cfg):
        self.cfg = cfg

    def retrieve_cfg(self):
        return self.cfg

class CameraJobCfg(JobCfg):
    def __init__(self, cfg: dict):
        # Modify cfg to include headless and enable_cameras
        cfg["runner_args"]["singletons"] = []
        cfg["runner_args"]["singletons"].append("--headless")
        cfg["runner_args"]["singletons"].append("--enable_cameras")
        cfg["workflow"] = "/workspace/isaaclab/workflows/rl_games/train.py"
        super().__init__(cfg)

class CameraJobCfgHelper(CameraJobCfg):
    """

    """
    def __init__(self, 
                    cfg = {},
                    vary_env_count: bool = True):
        if vary_env_count:
            cfg["runner_args"]["--num_envs"] = tune.randint(2**6, 2**14)
        pass # 


class CartpoleRGB(CameraJobCfgHelper):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-RGB-Camera-v0"
        super().__init__(cfg)

class CartpoleResNet(CameraJobCfgHelper):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-ResNet18-Camera-v0"
        super().__init__(cfg)

class CartpoleTheia(CameraJobCfgHelper):
    def __init__(self, cfg: dict = {}):
        cfg["runner_args"]["--task"] = "Isaac-Cartpole-TheiaTiny-Camera-v0"
        super().__init__(cfg)

def jobs():
    pass
