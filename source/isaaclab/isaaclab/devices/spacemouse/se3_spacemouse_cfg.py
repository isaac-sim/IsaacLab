# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SE(3) space mouse controller."""

from __future__ import annotations

from dataclasses import dataclass

from ..device_base import DeviceCfg
from .se3_spacemouse import Se3SpaceMouse


@dataclass
class Se3SpaceMouseCfg(DeviceCfg):
    """Configuration for SE3 space mouse devices."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    retargeters: None = None
    class_type: type | str = Se3SpaceMouse
