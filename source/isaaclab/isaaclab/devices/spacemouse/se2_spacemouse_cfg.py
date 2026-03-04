# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(2) space mouse controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..device_base import DeviceCfg

if TYPE_CHECKING:
    from .se2_spacemouse import Se2SpaceMouse


@configclass
class Se2SpaceMouseCfg(DeviceCfg):
    """Configuration for SE2 space mouse devices."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0
    class_type: type[Se2SpaceMouse] | str = "{DIR}.se2_spacemouse:Se2SpaceMouse"
