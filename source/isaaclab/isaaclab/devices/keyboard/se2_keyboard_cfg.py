# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(2) keyboard controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..device_base import DeviceCfg

if TYPE_CHECKING:
    from .se2_keyboard import Se2Keyboard


@configclass
class Se2KeyboardCfg(DeviceCfg):
    """Configuration for SE2 keyboard devices."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0
    class_type: type[Se2Keyboard] | str = "{DIR}.se2_keyboard:Se2Keyboard"
