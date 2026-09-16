# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(3) keyboard controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import config_field

from ..device_base import DeviceCfg

if TYPE_CHECKING:
    from .se3_keyboard import Se3Keyboard


@dataclass
class Se3KeyboardCfg(DeviceCfg):
    """Configuration for SE3 keyboard devices."""

    gripper_term: bool = config_field(True)
    pos_sensitivity: float = config_field(0.4)
    rot_sensitivity: float = config_field(0.8)
    retargeters: None = config_field(None)
    class_type: type[Se3Keyboard] | str = config_field("{DIR}.se3_keyboard:Se3Keyboard")
