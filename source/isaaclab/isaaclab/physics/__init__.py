# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation backends for simulation interfaces."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .physics_manager import PhysicsManager, PhysicsEvent, CallbackHandle
    from .physics_manager_cfg import PhysicsCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("physics_manager", ["PhysicsManager", "PhysicsEvent", "CallbackHandle"]),
    ("physics_manager_cfg", "PhysicsCfg"),
)
