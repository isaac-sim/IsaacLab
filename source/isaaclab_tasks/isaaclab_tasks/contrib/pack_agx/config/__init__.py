# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task assets and H2 + Sharpa embodiment exports for the AGX Orin packing task."""

from __future__ import annotations

import os

from isaaclab_tasks.contrib.h2_sharpa.camera_config import CameraPresets
from isaaclab_tasks.contrib.h2_sharpa.robot_config import H2RobotPresets

# Hosted in a private Hugging Face repository; direct URL access needs the repository to be public.
# Until then, point the environment variable at a local mirror that keeps the same Props/LightWheel/
# layout (the Isaac Healthcare bundle layout), e.g. a `snapshot_download` of the repo.
PACK_AGX_ASSET_ROOT: str = os.environ.get(
    "ISAACLAB_PACK_AGX_ASSET_ROOT",
    "https://huggingface.co/LiFanxing/isaaclab-rlinf-assets/resolve/main/Props/LightWheel",
)

TABLE_USD = f"{PACK_AGX_ASSET_ROOT}/Assets/Table256/Table256.usd"
AGX_ORIN_USD = f"{PACK_AGX_ASSET_ROOT}/Assets/MiniPc001/MiniPc001.usd"
PROTECTIVE_BOX_USD = f"{PACK_AGX_ASSET_ROOT}/Assets/ProtectiveBox001/ProtectiveBox001.usd"
BACKGROUND_USD = f"{PACK_AGX_ASSET_ROOT}/NuRec/IMG_6246_nurec_aligned_scaled.usdz"

__all__ = [
    "AGX_ORIN_USD",
    "BACKGROUND_USD",
    "CameraPresets",
    "H2RobotPresets",
    "PACK_AGX_ASSET_ROOT",
    "PROTECTIVE_BOX_USD",
    "TABLE_USD",
]
