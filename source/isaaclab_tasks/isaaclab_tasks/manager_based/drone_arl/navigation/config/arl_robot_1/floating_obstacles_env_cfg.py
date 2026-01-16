# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.drone_arl.navigation.config.arl_robot_1.navigation_env_cfg import (
    NavigationVelocityFloatingObstacleEnvCfg,
)

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


@configclass
class FloatingObstacleEnvCfg(NavigationVelocityFloatingObstacleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to arl_robot_1
        self.scene.robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt


@configclass
class FloatingObstacleEnvCfg_PLAY(FloatingObstacleEnvCfg):
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
