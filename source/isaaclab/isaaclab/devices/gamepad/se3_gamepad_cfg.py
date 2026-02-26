# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(3) gamepad controller."""

from __future__ import annotations

from dataclasses import dataclass

from ..device_base import DeviceCfg
from .se3_gamepad import Se3Gamepad


@dataclass
class Se3GamepadCfg(DeviceCfg):
    """Configuration for SE3 gamepad devices."""

    gripper_term: bool = True
    dead_zone: float = 0.01
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.6
    class_type: type | str = Se3Gamepad
