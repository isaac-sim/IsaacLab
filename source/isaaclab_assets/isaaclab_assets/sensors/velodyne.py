# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Velodyne LiDAR sensors."""

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

##
# Configuration
##

VELODYNE_VLP_16_RAYCASTER_CFG = RayCasterCfg(
    ray_alignment="base",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2
    ),
    debug_vis=True,
    max_distance=100,
)
"""Configuration for Velodyne Puck LiDAR (VLP-16) as a :class:`RayCasterCfg`.

Reference: https://velodynelidar.com/wp-content/uploads/2019/12/63-9229_Rev-K_Puck-_Datasheet_Web.pdf
"""
