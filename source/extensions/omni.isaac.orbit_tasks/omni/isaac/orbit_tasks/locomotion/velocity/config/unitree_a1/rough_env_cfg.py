# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit.assets.config.unitree import UNITREE_A1_CFG  # isort: skip


@configclass
class UnitreeA1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to unitree-a1
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        # override rewards
        # self.rewards.dof_torques_l2.weight = -2.0e-4
        # self.rewards.dof_pos_limits.weight = -10.0

        # disable pushing for now
        self.randomization.push_robot = None
        self.randomization.add_base_mass = None

        # change body and joint names
        # TODO: Change to .*foot once we make a new USD for the robot
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*calf"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*hip"]


@configclass
class UnitreeA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
