# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation backends for simulation interfaces."""

# from .physx_manager import PhysxManager, IsaacEvents
# from .physx_manager_cfg import PhysxManagerCfg
from .newton_manager import NewtonManager
from .newton_manager_cfg import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonSolverCfg, XPBDSolverCfg
from .physics_manager import CallbackHandle, PhysicsEvent, PhysicsManager
from .physics_manager_cfg import PhysicsCfg

__all__ = [
    "PhysicsManager",
    "PhysicsEvent",
    "PhysicsCfg",
    "CallbackHandle",
    "PhysicsManagerCfg",
    # "PhysxManager",
    # "IsaacEvents",
    # "PhysxManagerCfg",
    "NewtonManager",
    "NewtonCfg",
    "NewtonCfg",
    "NewtonSolverCfg",
    "MJWarpSolverCfg",
    "XPBDSolverCfg",
    "FeatherstoneSolverCfg",
]
