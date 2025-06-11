# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import scipy
import numpy as np
import math

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg




##
# Pre-defined configs
##
from isaaclab_assets.robots.alexander import ALEXANDER_V1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# @configclass
# class AlexanderRewardsCfg(RewardsCfg):
#     pass
    










@configclass
class AlexanderFlatVelocityEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Alexander flat mimic environment configuration."""

    # rewards: AlexanderRewardsCfg = AlexanderRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = ALEXANDER_V1.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/PELVIS_LINK"

        # actions
        self.actions.joint_pos.scale = 0.3

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "TORSO_LINK"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "TORSO_LINK"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        

        # rewards

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*PELVIS_LINK", ".*TORSO_LINK"]

        




        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # commands
        # self.commands.base_velocity.root_animation = self.root_animation









@configclass
class AlexanderFlatVelocityEnvCfg_PLAY(AlexanderFlatVelocityEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # general settings
        # self.decimation = 1
        # self.episode_length_s = 10
        # # simulation settings
        # self.sim.dt = 0.00000001

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2
        self.episode_length_s = 100
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


