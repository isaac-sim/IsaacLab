# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(2) gamepad controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..device_base import DeviceCfg

if TYPE_CHECKING:
    from .se2_gamepad import Se2Gamepad


@configclass
class Se2GamepadCfg(DeviceCfg):
    """Configuration for SE2 gamepad devices."""

    v_x_sensitivity: float = 1.0
    v_y_sensitivity: float = 1.0
    omega_z_sensitivity: float = 1.0
    dead_zone: float = 0.01
    class_type: type[Se2Gamepad] | str = "{DIR}.se2_gamepad:Se2Gamepad"
