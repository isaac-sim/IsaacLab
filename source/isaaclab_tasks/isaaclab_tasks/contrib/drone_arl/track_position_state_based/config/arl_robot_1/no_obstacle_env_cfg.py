# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

from .track_position_state_based_env_cfg import TrackPositionNoObstaclesEnvCfg

##
# Pre-defined configs
##


@configclass
class NoObstacleEnvCfg(TrackPositionNoObstaclesEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to arl_robot_1
        self.scene.robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

    def play_mode(self):
        # play-mode overrides of parent
        super().play_mode()

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
