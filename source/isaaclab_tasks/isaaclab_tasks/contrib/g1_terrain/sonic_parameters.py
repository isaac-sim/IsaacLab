# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor and observation ordering for the released SONIC G1 models."""

import numpy as np

JOINT_NAMES = (
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
# G1_ISAACLab_ORDER in gear_sonic/envs/env_utils/joint_utils.py:
# https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
POLICY_ORDER = np.array([JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES])
DEFAULT_POS = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0] * 2
    + [0.0] * 3
    + [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
    + [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_ARMATURE = np.array(
    [0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.003609725, 0.003609725] * 2
    + [0.010177520, 0.003609725, 0.003609725]
    + [0.003609725] * 5
    + [0.00425] * 2
    + [0.003609725] * 5
    + [0.00425] * 2
)
_EFFORT = np.array([139, 139, 88, 139, 25, 25] * 2 + [88, 25, 25] + ([25] * 5 + [5] * 2) * 2)
_OMEGA = 20.0 * 3.1415926535
ACTION_SCALE = (0.25 * _EFFORT / (_ARMATURE * _OMEGA**2)).astype(np.float32)
_GAIN_MULTIPLIER = np.array([1, 1, 1, 1, 2, 2] * 2 + [1, 2, 2] + [1] * 14)
STIFFNESS = (_GAIN_MULTIPLIER * _ARMATURE * _OMEGA**2).astype(np.float32)
DAMPING = (4.0 * _GAIN_MULTIPLIER * _ARMATURE * _OMEGA).astype(np.float32)
