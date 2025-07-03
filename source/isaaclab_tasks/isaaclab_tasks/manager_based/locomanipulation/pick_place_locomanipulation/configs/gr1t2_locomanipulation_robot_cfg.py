# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for GR1T2 locomanipulation robot.

This module provides configurations for the GR1T2 humanoid robot specifically
designed for locomanipulation tasks, including both fixed base and mobile
configurations for upper body manipulation.
"""

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip


##
# Configuration for GR1T2 Locomanipulation
##

GR1T2_LOCOMANIPULATION_ROBOT_CFG = GR1T2_CFG.replace(
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.93),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            # right-arm
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": -1.5708,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            # left-arm
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": -1.5708,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            # --
            "head_.*": 0.0,
            "waist_.*": 0.0,
            ".*_hip_.*": 0.0,
            ".*_knee_.*": 0.0,
            ".*_ankle_.*": 0.0,
            "R_.*": 0.0,
            "L_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
)
