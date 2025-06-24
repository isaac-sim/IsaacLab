# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.so_100 import SO_100_HIGH_PD_CFG  # isort: skip


@configclass
class SO100CubeStackIKRelEnvCfg(stack_joint_pos_env_cfg.SO100CubeStackJointPosEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = SO_100_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                      init_state=ArticulationCfg.InitialStateCfg(
                                                        pos=(0, 0, 0.0),
                                                        # rot=(0.7071, 0, 0, 0.7071),
                                                        rot=(1.0, 0, 0, 0.0),
                                                        joint_pos={
                                                            # right-arm
                                                            "a_1": 0.0,
                                                            "a_2": 1.5708,
                                                            "a_3": -1.5708,
                                                            "a_4": 1.2,
                                                            "a_5": 0.0,
                                                        },
                                                        joint_vel={".*": 0.0},
                                                    ))

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["a_[1-4]"],
            body_name="wrist",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
