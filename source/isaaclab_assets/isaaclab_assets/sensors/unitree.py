# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for cameras integrated into Unitree robots."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

##
# Configuration
##

H2_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/head_yaw_link/head_camera",
    update_period=0.02,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=0.2667,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_nominal_width=640.0,
        fisheye_nominal_height=480.0,
        fisheye_optical_centre_x=313.66428,
        fisheye_optical_centre_y=238.71598,
        fisheye_max_fov=140,
        fisheye_polynomial_a=0.0,
        fisheye_polynomial_b=3.389688850346405e-3,
        fisheye_polynomial_c=-3.917026323653134e-7,
        fisheye_polynomial_d=7.433522068283752e-9,
        fisheye_polynomial_e=-2.3148518556166975e-11,
        fisheye_polynomial_f=5.110486752596073e-14,
        clipping_range=(0.1, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        # Unitree H2 URDF camera mount; rpy=(0, 0, 0) converted to OpenGL xyzw.
        pos=(0.08667, 0.03, 0.0099),
        rot=(0.5, -0.5, -0.5, 0.5),
        convention="opengl",
    ),
)
"""Head camera built into the Unitree H2. The wrist cameras are retrofitted, so their mounts
live with the task that installs them."""
