# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Sterolabs Depth Cameras."""


from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, patterns
from omni.isaac.lab.sim.spawners import PinholeCameraCfg

##
# Configuration as RayCasterCameraCfg
##

ZED_X_NARROW = RayCasterCameraCfg(
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=4.0,
        intrinsic_matrix=[380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    debug_vis=True,
    max_distance=20,
    data_types=["distance_to_image_plane"],
)

ZED_X_MINI_WIDE = RayCasterCameraCfg(
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=2.2,
        intrinsic_matrix=[369.7771, 0.0, 489.9926, 0.0, 369.7771, 275.9385, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    debug_vis=True,
    max_distance=20,
    data_types=["distance_to_image_plane"],
)


##
# Configuration as USD Camera
##

ZED_X_NARROW_USD = CameraCfg(
    spawn=PinholeCameraCfg(
        focal_length=4.0,
        intrinsic_matrix=[380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0],
    ),
    height=540,
    width=960,
    data_types=["distance_to_image_plane", "rgb"],
)

ZED_X_MINI_WIDE_USD = CameraCfg(
    spawn=PinholeCameraCfg(
        focal_length=2.2,
        intrinsic_matrix=[369.7771, 0.0, 489.9926, 0.0, 369.7771, 275.9385, 0.0, 0.0, 1.0],
    ),
    height=540,
    width=960,
    data_types=["distance_to_image_plane", "rgb"],
)
