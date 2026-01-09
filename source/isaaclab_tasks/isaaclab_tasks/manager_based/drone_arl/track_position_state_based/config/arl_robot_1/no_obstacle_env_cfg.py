# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

from isaaclab.utils import configclass

<<<<<<<< HEAD:source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/track_position_state_based/config/arl_robot_1/multirotor_track_position_state_based_env_cfg.py
from .track_position_state_based_env_cfg import TrackPositionStateBasedEnvCfg
========
from .state_based_control_env_cfg import TrackPositionNoObstaclesEnvCfg
>>>>>>>> 42abd741447835a6c3449161804a582fa1d12bd2:source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/track_position_state_based/config/arl_robot_1/no_obstacle_env_cfg.py

##
# Pre-defined configs
##


@configclass
<<<<<<<< HEAD:source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/track_position_state_based/config/arl_robot_1/multirotor_track_position_state_based_env_cfg.py
class MultirotorTrackPositionStateBasedEnvCfg(TrackPositionStateBasedEnvCfg):
========
class NoObstacleEnvCfg(TrackPositionNoObstaclesEnvCfg):
>>>>>>>> 42abd741447835a6c3449161804a582fa1d12bd2:source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/track_position_state_based/config/arl_robot_1/no_obstacle_env_cfg.py
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to arl_robot_1
        self.scene.robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt


@configclass
<<<<<<<< HEAD:source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/track_position_state_based/config/arl_robot_1/multirotor_track_position_state_based_env_cfg.py
class MultirotorTrackPositionStateBasedEnvCfg_PLAY(MultirotorTrackPositionStateBasedEnvCfg):
========
class NoObstacleEnvCfg_PLAY(NoObstacleEnvCfg):
>>>>>>>> 42abd741447835a6c3449161804a582fa1d12bd2:source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/track_position_state_based/config/arl_robot_1/no_obstacle_env_cfg.py
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
