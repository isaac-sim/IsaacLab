# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Sterolabs Depth Cameras.

The following configuration parameters are available:

* :obj:`ZED_X_NARROW_RAYCASTER_CFG`: The ZED X Camera with narrow field of view as RayCasterCameraCfg
* :obj:`ZED_X_MINI_WIDE_RAYCASTER_CFG`: The ZED X Mini Camera with wide field of view as RayCasterCameraCfg
* :obj:`ZED_X_NARROW_USD_CFG`: The ZED X Camera with narrow field of view as CameraCfg
* :obj:`ZED_X_MINI_WIDE_USD_CFG`: The ZED X Mini Camera with wide field of view as CameraCfg

Reference:

* https://store.stereolabs.com/cdn/shop/files/ZED_X_and_ZED_X_Mini_Datasheet.pdf?v=10269370652353745680

"""


from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, patterns
from omni.isaac.lab.sim.spawners import PinholeCameraCfg

##
# Configuration as RayCasterCameraCfg
##

ZED_X_NARROW_RAYCASTER_CFG = RayCasterCameraCfg(
    pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
        focal_length=38.0,
        intrinsic_matrix=[380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    debug_vis=True,
    max_distance=20,
    data_types=["distance_to_image_plane"],
)

ZED_X_MINI_WIDE_RAYCASTER_CFG = RayCasterCameraCfg(
    pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
        focal_length=22.0,
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

ZED_X_NARROW_USD_CFG = CameraCfg(
    spawn=PinholeCameraCfg.from_intrinsic_matrix(
        focal_length=38.0,
        intrinsic_matrix=[380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    height=540,
    width=960,
    data_types=["distance_to_image_plane", "rgb"],
)

ZED_X_MINI_WIDE_USD_CFG = CameraCfg(
    spawn=PinholeCameraCfg.from_intrinsic_matrix(
        focal_length=22.0,
        intrinsic_matrix=[369.7771, 0.0, 489.9926, 0.0, 369.7771, 275.9385, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    height=540,
    width=960,
    data_types=["distance_to_image_plane", "rgb"],
)
