# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for pink controller.

This module provides configurations for humanoid robot pink IK controllers,
including both fixed base and mobile configurations for upper body manipulation.
"""

from pink.tasks import FrameTask

from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

##
# Pink IK Controller Configuration for G1
##

G1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=14,
    show_ik_warnings=False,
    variable_input_tasks=[
        FrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.8,
        ),
        FrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.8,
        ),
    ],
    fixed_input_tasks=[],
)
"""Base configuration for the G1 pink IK controller.

This configuration sets up the pink IK controller for the G1 humanoid robot with
left and right wrist control tasks. The controller is designed for upper body
manipulation tasks.
"""


##
# Pink IK Action Configuration for G1
##

G1_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
    ],
    # Fixed joints for IK (pelvis, legs, and hands are fixed)
    ik_urdf_fixed_joint_names=[
        ".*_hip_yaw_joint",
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*waist.*",
        ".*_knee_joint",
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
        ".*_index_.*",
        ".*_middle_.*",
        ".*_thumb_.*",
    ],
    hand_joint_names=[
        "left_hand_index_0_joint",      # Index finger proximal
        "left_hand_middle_0_joint",     # Middle finger proximal
        "left_hand_thumb_0_joint",      # Thumb base (yaw axis)
        "right_hand_index_0_joint",     # Index finger proximal
        "right_hand_middle_0_joint",    # Middle finger proximal
        "right_hand_thumb_0_joint",     # Thumb base (yaw axis)
        "left_hand_index_1_joint",      # Index finger distal
        "left_hand_middle_1_joint",     # Middle finger distal
        "left_hand_thumb_1_joint",      # Thumb middle (pitch axis)
        "right_hand_index_1_joint",     # Index finger distal
        "right_hand_middle_1_joint",    # Middle finger distal
        "right_hand_thumb_1_joint",      # Thumb middle (pitch axis)
        "left_hand_thumb_2_joint",      # Thumb tip
        "right_hand_thumb_2_joint",     # Thumb tip
    ],
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    # Configuration for the IK controller
    # The frames names are the ones present in the URDF file
    # The urdf has to be generated from the USD that is being used in the scene
    controller=G1_UPPER_BODY_IK_CONTROLLER_CFG,
)
"""Base configuration for the G1 pink IK action.

This configuration sets up the pink IK action for the G1 humanoid robot,
defining which joints are controlled by the IK solver and which are fixed.
The configuration includes:
- Upper body joints controlled by IK (shoulders, elbows, wrists)
- Fixed joints (pelvis, legs, hands)
- Hand joint names for additional control
- Reference to the pink IK controller configuration
"""


##
# Pink IK Controller Configuration for GR1T2
##

GR1T2_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="base_link",
    num_hand_joints=22,
    show_ik_warnings=False,
    variable_input_tasks=[
        FrameTask(
            "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.1,
        ),
        FrameTask(
            "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.1,
        ),
    ],
    fixed_input_tasks=[],
)
"""Base configuration for the GR1T2 pink IK controller.

This configuration sets up the pink IK controller for the GR1T2 humanoid robot with
left and right hand control tasks. The controller is designed for upper body
manipulation tasks with fixed base.
"""


##
# Pink IK Action Configuration for GR1T2
##

GR1T2_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_wrist_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
    ],
    ik_urdf_fixed_joint_names=[
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_pitch_joint",
        "right_knee_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "L_index_proximal_joint",
        "L_middle_proximal_joint",
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "L_index_intermediate_joint",
        "L_middle_intermediate_joint",
        "L_pinky_intermediate_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_pitch_joint",
        "R_index_intermediate_joint",
        "R_middle_intermediate_joint",
        "R_pinky_intermediate_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_pitch_joint",
        "L_thumb_distal_joint",
        "R_thumb_distal_joint",
        "head_roll_joint",
        "head_pitch_joint",
        "head_yaw_joint",
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
    ],
    hand_joint_names=[
        "L_index_proximal_joint",
        "L_middle_proximal_joint",
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "L_index_intermediate_joint",
        "L_middle_intermediate_joint",
        "L_pinky_intermediate_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_pitch_joint",
        "R_index_intermediate_joint",
        "R_middle_intermediate_joint",
        "R_pinky_intermediate_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_pitch_joint",
        "L_thumb_distal_joint",
        "R_thumb_distal_joint",
    ],
    target_eef_link_names={
        "left_wrist": "left_hand_pitch_link",
        "right_wrist": "right_hand_pitch_link",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    # Configuration for the IK controller
    # The frames names are the ones present in the URDF file
    # The urdf has to be generated from the USD that is being used in the scene
    controller=GR1T2_UPPER_BODY_IK_CONTROLLER_CFG,
)
"""Base configuration for the GR1T2 pink IK action.

This configuration sets up the pink IK action for the GR1T2 humanoid robot,
defining which joints are controlled by the IK solver and which are fixed.
The configuration includes:
- Upper body joints controlled by IK (shoulders, elbows, wrists)
- Fixed joints (pelvis, legs, hands, head, waist)
- Hand joint names for additional control
- Reference to the pink IK controller configuration
"""
