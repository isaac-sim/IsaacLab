# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sharpa Wave wrist cameras retrofitted onto the Unitree H2.

The head camera ships with the robot and lives in :mod:`isaaclab_assets.sensors.unitree`; the wrist
brackets are task hardware, so their mounts and calibration stay with the task. Both are authored at
the calibration resolution and resized per task with ``.replace(height=..., width=...)``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_assets.sensors.unitree import H2_HEAD_CAMERA_CFG

from .metadata import CAMERA_BY_NAME

##
# Configuration
##

LEFT_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/left_hand_C_MC/left_wrist_camera",
    update_period=0.02,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=0.15435,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_nominal_width=640.0,
        fisheye_nominal_height=480.0,
        fisheye_optical_centre_x=319.90624,
        fisheye_optical_centre_y=239.67252,
        fisheye_max_fov=196,
        fisheye_polynomial_a=0.0,
        fisheye_polynomial_b=5.788058552990647e-3,
        fisheye_polynomial_c=1.6881056765838268e-6,
        fisheye_polynomial_d=-4.2228624221772085e-8,
        fisheye_polynomial_e=1.4575948452911756e-10,
        fisheye_polynomial_f=-1.4645006973842839e-13,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.07498591, 0.0007267178, 0.004823103),
        rot=(-0.6486821, 0.6996208, 0.1177079, 0.2754761),
        convention="opengl",
    ),
)
"""Left Sharpa Wave wrist camera, mounted on the hand's CAD bracket."""

RIGHT_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/right_hand_C_MC/right_wrist_camera",
    update_period=0.02,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=0.15435,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_nominal_width=640.0,
        fisheye_nominal_height=480.0,
        fisheye_optical_centre_x=319.90624,
        fisheye_optical_centre_y=239.67252,
        fisheye_max_fov=196,
        fisheye_polynomial_a=0.0,
        fisheye_polynomial_b=5.788058552990647e-3,
        fisheye_polynomial_c=1.6881056765838268e-6,
        fisheye_polynomial_d=-4.2228624221772085e-8,
        fisheye_polynomial_e=1.4575948452911756e-10,
        fisheye_polynomial_f=-1.4645006973842839e-13,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.07964737, 0.001976802, 0.01021968),
        rot=(0.6884326, -0.664184, 0.2683462, 0.1136246),
        convention="opengl",
    ),
)
"""Right Sharpa Wave wrist camera, mirrored from the left mount."""

FRONT_CAMERA_CFG = H2_HEAD_CAMERA_CFG
"""Head camera. It ships with the robot, so the cfg lives in :mod:`isaaclab_assets.sensors.unitree`."""

CAMERAS: dict[str, CameraCfg] = {
    "front_camera": FRONT_CAMERA_CFG,
    "left_wrist_camera": LEFT_WRIST_CAMERA_CFG,
    "right_wrist_camera": RIGHT_WRIST_CAMERA_CFG,
}
"""Every camera the policy reads, keyed by the camera names in :mod:`.metadata`."""

assert set(CAMERAS) == set(CAMERA_BY_NAME), (
    f"CAMERAS {set(CAMERAS)} out of sync with metadata cameras {set(CAMERA_BY_NAME)}"
)
