# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad device for SE(2) and SE(3) control."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .se2_gamepad import Se2Gamepad
    from .se2_gamepad_cfg import Se2GamepadCfg
    from .se3_gamepad import Se3Gamepad
    from .se3_gamepad_cfg import Se3GamepadCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("se2_gamepad", "Se2Gamepad"),
    ("se2_gamepad_cfg", "Se2GamepadCfg"),
    ("se3_gamepad", "Se3Gamepad"),
    ("se3_gamepad_cfg", "Se3GamepadCfg"),
)
