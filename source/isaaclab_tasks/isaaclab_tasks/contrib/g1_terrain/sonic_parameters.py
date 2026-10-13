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
POLICY_ORDER = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
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
