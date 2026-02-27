# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation backends for simulation interfaces."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .physx_manager import PhysxManager, IsaacEvents
    from .physx_manager_cfg import PhysxCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("physx_manager", ["PhysxManager", "IsaacEvents"]),
    ("physx_manager_cfg", "PhysxCfg"),
)
