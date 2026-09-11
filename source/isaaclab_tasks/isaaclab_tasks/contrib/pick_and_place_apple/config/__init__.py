# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task assets and H2 + Sharpa embodiment exports for the pick-and-place apple task."""

from __future__ import annotations

import os

from isaaclab_tasks.contrib.h2_sharpa.camera_config import CameraBaseCfg, CameraPresets
from isaaclab_tasks.contrib.h2_sharpa.metadata import H2_DEFAULT_JOINT_POS, H2_PNP_APPLE_CUSTOM_JOINT_POS
from isaaclab_tasks.contrib.h2_sharpa.robot_config import H2RobotPresets, make_h2_sharpa_cfg

# Hosted in a private Hugging Face dataset repository; direct URL access needs the repository to be
# public. Until then, point the environment variable at a local mirror that keeps the same
# Props/LightWheel/ layout (the Isaac Healthcare bundle layout), e.g. a `snapshot_download` of the repo.
PICK_AND_PLACE_APPLE_ASSET_ROOT: str = os.environ.get(
    "ISAACLAB_PICK_AND_PLACE_APPLE_ASSET_ROOT",
    "https://huggingface.co/datasets/LiFanxing/IsaacLabRLinfDemo/resolve/main/assets/Props/LightWheel",
)

TABLE_USD = f"{PICK_AND_PLACE_APPLE_ASSET_ROOT}/Assets/Table256/Table256_cloth.usd"
APPLE_USD = f"{PICK_AND_PLACE_APPLE_ASSET_ROOT}/Assets/Apple033/Apple033.usd"
PLATE_USD = f"{PICK_AND_PLACE_APPLE_ASSET_ROOT}/Assets/SimReady_Furniture/plate_large/plate_large_rigid.usd"
BACKGROUND_USD = f"{PICK_AND_PLACE_APPLE_ASSET_ROOT}/NuRec/IMG_6246_nurec_aligned_scaled.usdz"

# Task-specific start pose for the apple task.
H2_PNP_APPLE_INIT_POS: tuple[float, float, float] = (-0.95, 0.0, 1.05)
H2_PNP_APPLE_INIT_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

__all__ = [
    "APPLE_USD",
    "BACKGROUND_USD",
    "CameraBaseCfg",
    "CameraPresets",
    "H2_DEFAULT_JOINT_POS",
    "H2_PNP_APPLE_CUSTOM_JOINT_POS",
    "H2_PNP_APPLE_INIT_POS",
    "H2_PNP_APPLE_INIT_ROT",
    "H2RobotPresets",
    "PICK_AND_PLACE_APPLE_ASSET_ROOT",
    "PLATE_USD",
    "TABLE_USD",
    "make_h2_sharpa_cfg",
]
