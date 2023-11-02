# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_envs.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# isort: off
from omni.isaac.orbit.assets.config.unitree import UNITREE_A1_CFG


@configclass
class UnitreeA1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to unitree a1
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 1.0
        # self.rewards.dof_torques_l2.weight = -2.0e-4
        # self.rewards.dof_pos_limits.weight = -10.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # disable pushing for now
        self.randomization.push_robot = None
        self.randomization.add_base_mass = None

        # change body and joint names
        # TODO: Change to .*foot once we make a new USD for the robot
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*calf"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*hip"]


class UnitreeA1FlatEnvCfg_PLAY(UnitreeA1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
