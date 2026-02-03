# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation backends for simulation interfaces."""

# from .newton_backend import NewtonBackend
from .physics_manager import PhysicsManager
from .physics_manager_cfg import PhysicsManagerCfg
from .physx_manager import PhysxManager, IsaacEvents
from .physx_manager_cfg import PhysxManagerCfg

__all__ = [
    # "NewtonBackend",
    "PhysicsManager",
    "PhysicsManagerCfg",
    "PhysxManager",
    "PhysxManagerCfg",
    "IsaacEvents",
]
