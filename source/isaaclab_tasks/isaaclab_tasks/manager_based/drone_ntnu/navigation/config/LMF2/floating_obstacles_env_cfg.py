# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.drone_ntnu.navigation.config.LMF2.navigation_env_cfg import NavigationVelocityFloatingObstacleEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.lmf2 import LMF2_CFG


@configclass
class LMF2FloatingObstacleEnvCfg(NavigationVelocityFloatingObstacleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to lmf2
        self.scene.robot = LMF2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class LMF2FloatingObstacleEnvCfg_PLAY(LMF2FloatingObstacleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.curriculum.obstacle_levels.params["max_difficulty"] = 40
        self.curriculum.obstacle_levels.params["min_difficulty"] = 39
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
