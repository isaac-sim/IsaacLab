# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from pink.tasks import FrameTask


from . import stack_joint_pos_env_cfg

import tempfile

##
# Pre-defined configs
##
from isaaclab_assets.robots.so_100 import SO_100_CFG  # isort: skip


@configclass
class SO100CubeStackPinkIKAbsEnvCfg(stack_joint_pos_env_cfg.SO100CubeStackJointPosEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Temporary directory for URDF files
        self.temp_urdf_dir = tempfile.gettempdir()
        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = SO_100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                      init_state=ArticulationCfg.InitialStateCfg(
                                                        pos=(0, 0, 0.0),
                                                        rot=(0.7071, 0, 0, 0.7071),
                                                        joint_pos={
                                                            # right-arm
                                                            "shoulder_pan_joint": 0.0,
                                                            "shoulder_lift_joint": 1.5708,
                                                            "elbow_flex_joint": -1.5708,
                                                            "wrist_flex_joint": 1.2,
                                                            "wrist_roll_joint": 0.0,
                                                            "gripper_joint": 0.0,
                                                        },
                                                        joint_vel={".*": 0.0},
                                                    ))

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_flex_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ],
            # Joints to be locked in URDF
            ik_urdf_fixed_joint_names=["gripper_joint"],
            hand_joint_names=[],
            # the robot in the sim scene we are controlling
            asset_name="robot",
            # Configuration for the IK controller
            # The frames names are the ones present in the URDF file
            # The urdf has to be generated from the USD that is being used in the scene
            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="base",
                num_hand_joints=0,
                show_ik_warnings=True,
                variable_input_tasks=[
                    FrameTask(
                        "gripper",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.1,
                    )
                ],
                fixed_input_tasks=[
                    # COMMENT OUT IF LOCKING WAIST/HEAD
                    # FrameTask(
                    #     "GR1T2_fourier_hand_6dof_head_yaw_link",
                    #     position_cost=1.0,  # [cost] / [m]
                    #     orientation_cost=0.05,  # [cost] / [rad]
                    # ),
                ],
            ),
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.arm_action.ik_urdf_fixed_joint_names
        )
    
        # Set the URDF and mesh paths for the IK controller
        self.actions.arm_action.controller.urdf_path = temp_urdf_output_path
        self.actions.arm_action.controller.mesh_path = temp_urdf_meshes_output_path

