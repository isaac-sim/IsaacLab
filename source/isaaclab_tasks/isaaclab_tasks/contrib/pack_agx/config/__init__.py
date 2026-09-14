# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task assets and H2 + Sharpa embodiment exports for the AGX Orin packing task."""

from __future__ import annotations

from isaaclab_tasks.contrib.rlinf_assets import NUREC_ASSET_ROOT, PROP_ASSET_ROOT
from ..camera_config import CameraPresets
from ..robot_config import H2RobotPresets

TABLE_USD = f"{PROP_ASSET_ROOT}/Assets/Table256/Table256.usd"
AGX_ORIN_USD = f"{PROP_ASSET_ROOT}/Assets/MiniPc001/MiniPc001.usd"
PROTECTIVE_BOX_USD = f"{PROP_ASSET_ROOT}/Assets/ProtectiveBox001/ProtectiveBox001.usd"
BACKGROUND_USD = f"{NUREC_ASSET_ROOT}/IMG_6246_nurec_aligned_scaled.usdz"

__all__ = [
    "AGX_ORIN_USD",
    "BACKGROUND_USD",
    "CameraPresets",
    "H2RobotPresets",
    "PROTECTIVE_BOX_USD",
    "TABLE_USD",
]
