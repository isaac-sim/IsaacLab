# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(3) gamepad controller."""

from __future__ import annotations

from ...utils import configclass
from ..device_base import DeviceCfg


@configclass
class Se3GamepadCfg(DeviceCfg):
    """Configuration translated to Isaac Capture before device creation.

    .. deprecated:: 3.2
        Use :func:`isaaclab_teleop.se3_gamepad_teleop_cfg` for new configurations.
    """

    gripper_term: bool = True
    dead_zone: float = 0.01
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.6
