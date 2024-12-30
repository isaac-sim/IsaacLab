# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Velodyne LiDAR sensors."""


from isaaclab.sensors import RayCasterCfg, patterns

##
# Configuration
##

VELODYNE_VLP_16_RAYCASTER_CFG = RayCasterCfg(
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2
    ),
    debug_vis=True,
    max_distance=100,
)
"""Configuration for Velodyne Puck LiDAR (VLP-16) as a :class:`RayCasterCfg`.

Reference: https://velodynelidar.com/wp-content/uploads/2019/12/63-9229_Rev-K_Puck-_Datasheet_Web.pdf
"""
