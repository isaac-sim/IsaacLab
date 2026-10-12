# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment parameters for the G1 depth-based locomotion policy."""

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
            "hand_index_0",
            "hand_index_1",
            "hand_middle_0",
            "hand_middle_1",
            "hand_thumb_0",
            "hand_thumb_1",
            "hand_thumb_2",
        )
    ]
)
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    ".*_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
}
ACTION_SCALE = {
    ".*_hip_yaw_joint": 0.1467,
    ".*_hip_roll_joint": 0.1467,
    ".*_hip_pitch_joint": 0.11,
    ".*_knee_joint": 0.1738,
    "waist_yaw_joint": 0.11,
    "waist_roll_joint": 0.0625,
    "waist_pitch_joint": 0.0625,
    ".*_ankle_pitch_joint": 0.625,
    ".*_ankle_roll_joint": 0.625,
    ".*_shoulder_.*_joint": 0.1563,
    ".*_elbow_joint": 0.1563,
    ".*_wrist_roll_joint": 0.1563,
    ".*_wrist_pitch_joint": 0.0313,
    ".*_wrist_yaw_joint": 0.0313,
    ".*_hand_.*_joint": 0.0153,
}
HISTORY_LENGTH = 5
DEPTH_SHAPE = (3, 38, 64)
DEPTH_MAX_RANGE = 3.0
