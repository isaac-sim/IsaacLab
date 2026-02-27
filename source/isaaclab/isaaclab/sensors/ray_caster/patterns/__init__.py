# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for ray-casting patterns used by the ray-caster."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .patterns import bpearl_pattern, grid_pattern, lidar_pattern, pinhole_camera_pattern
    from .patterns_cfg import BpearlPatternCfg, GridPatternCfg, LidarPatternCfg, PatternBaseCfg, PinholeCameraPatternCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("patterns", [
        "bpearl_pattern",
        "grid_pattern",
        "lidar_pattern",
        "pinhole_camera_pattern",
    ]),
    ("patterns_cfg", [
        "BpearlPatternCfg",
        "GridPatternCfg",
        "LidarPatternCfg",
        "PatternBaseCfg",
        "PinholeCameraPatternCfg",
    ]),
)
