# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package providing IsaacTeleop-based teleoperation for Isaac Lab."""

from __future__ import annotations

import os
import toml
ISAACLAB_TELEOP_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
ISAACLAB_TELEOP_METADATA = toml.load(os.path.join(ISAACLAB_TELEOP_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
__version__ = ISAACLAB_TELEOP_METADATA["package"]["version"]

import typing

if typing.TYPE_CHECKING:
    from .isaac_teleop_cfg import IsaacTeleopCfg
    from .isaac_teleop_device import IsaacTeleopDevice, create_isaac_teleop_device
    from .xr_anchor_utils import XrAnchorSynchronizer
    from .xr_cfg import XrAnchorRotationMode, XrCfg, remove_camera_configs

from isaaclab.utils.module import lazy_export

lazy_export(
    ("isaac_teleop_cfg", "IsaacTeleopCfg"),
    ("isaac_teleop_device", ["IsaacTeleopDevice", "create_isaac_teleop_device"]),
    ("xr_anchor_utils", "XrAnchorSynchronizer"),
    ("xr_cfg", ["XrAnchorRotationMode", "XrCfg", "remove_camera_configs"]),
)
