# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predefined configurations for GelSight tactile sensors."""

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg

##
# Predefined Configurations
##

GELSIGHT_R15_CFG = GelSightRenderCfg(
    sensor_data_dir_name="gelsight_r15_data",
    background_path="bg.jpg",
    calib_path="polycalib.npz",
    real_background="real_bg.npy",
    image_height=320,
    image_width=240,
    num_bins=120,
    mm_per_pixel=0.0877,
)
"""Configuration for GelSight R1.5 sensor rendering parameters.

The GelSight R1.5 is a high-resolution tactile sensor with a 320x240 pixel tactile image.
It uses a pixel-to-millimeter ratio of 0.0877 mm/pixel.

Reference: https://www.gelsight.com/gelsightinc-products/
"""

GELSIGHT_MINI_CFG = GelSightRenderCfg(
    sensor_data_dir_name="gs_mini_data",
    background_path="bg.jpg",
    calib_path="polycalib.npz",
    real_background="real_bg.npy",
    image_height=240,
    image_width=320,
    num_bins=120,
    mm_per_pixel=0.065,
)
"""Configuration for GelSight Mini sensor rendering parameters.

The GelSight Mini is a compact tactile sensor with a 240x320 pixel tactile image.
It uses a pixel-to-millimeter ratio of 0.065 mm/pixel, providing higher spatial resolution
than the R1.5 model.

Reference: https://www.gelsight.com/gelsightinc-products/
"""
