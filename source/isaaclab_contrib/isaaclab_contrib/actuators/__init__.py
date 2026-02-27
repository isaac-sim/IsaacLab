# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for thruster actuator models.

This package provides actuator models specifically designed for multirotor thrusters.
The thruster actuator simulates realistic motor/propeller dynamics including asymmetric
rise and fall time constants, thrust limits, and dynamic response characteristics.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .thruster import Thruster
    from .thruster_cfg import ThrusterCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("thruster", "Thruster"),
    ("thruster_cfg", "ThrusterCfg"),
)
