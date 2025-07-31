# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, SurfaceGripperBinaryActionCfg
from isaaclab.utils import configclass
from isaaclab.assets import SurfaceGripperCfg

from . import stack_joint_pos_env_cfg



@configclass
class UR10ParallelGripperCubeStackEnvCfg(stack_joint_pos_env_cfg.UR10ParallelGripperCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (ur10)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.06]),
        )



@configclass
class UR10LongSuctionCubeStackEnvCfg(stack_joint_pos_env_cfg.UR10LongSuctionCubeStackEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (ur10)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_suction_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )



@configclass
class UR10ShortSuctionCubeStackEnvCfg(stack_joint_pos_env_cfg.UR10ShortSuctionCubeStackEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (UR10 SHORT SUCTION)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
        )

