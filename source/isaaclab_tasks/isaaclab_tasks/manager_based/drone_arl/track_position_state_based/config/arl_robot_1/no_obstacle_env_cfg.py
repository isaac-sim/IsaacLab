# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

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


@configclass
class NoObstacleEnvCfg_PLAY(NoObstacleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
