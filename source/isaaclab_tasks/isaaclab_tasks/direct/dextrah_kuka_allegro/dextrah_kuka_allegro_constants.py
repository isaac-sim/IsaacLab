# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

# Constants (RARELY CHANGE)
XYZ_LIST = ["x", "y", "z"]
RPY_LIST = ["R", "P", "Y"]
NUM_XYZ = len(XYZ_LIST)
NUM_RPY = len(RPY_LIST)
NUM_QUAT = 4



HAND_PCA_MINS = [0.2475, -0.3286, -0.7238, -0.0192, -0.5532]
HAND_PCA_MAXS = [3.8336, 3.0025, 0.8977, 1.0243, 0.0629]
#PALM_POSE_MINS = [-1, -0.75, 0, -np.pi, -np.pi / 2, -np.pi]
#PALM_POSE_MAXS = [0.25, 0.75, 1, np.pi, np.pi / 2, np.pi]

deg2rad = np.pi / 180.
def PALM_POSE_MINS_FUNC(max_pose_angle):
    return [
        -1.2, -0.7, 0.,
        (-135. - max_pose_angle) * deg2rad,
        -max_pose_angle * deg2rad,
        (180. - max_pose_angle) * deg2rad
    ]


def PALM_POSE_MAXS_FUNC(max_pose_angle):
    return [
        0., 0.7, 1.,
        (-135. + max_pose_angle) * deg2rad,
        max_pose_angle * deg2rad,
        (180. + max_pose_angle) * deg2rad
    ]

NUM_HAND_PCA = 5

#TABLE_LENGTH_X, TABLE_LENGTH_Y, TABLE_LENGTH_Z = 0.725, 1.16, 0.03
