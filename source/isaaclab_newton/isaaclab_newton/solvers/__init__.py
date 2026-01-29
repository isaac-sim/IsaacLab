# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Solvers module for position-based dynamics simulations.

This module contains implementations of various constraint solvers,
including the Direct Position-Based Solver for Stiff Rods based on
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods".

Features:
- XPBD (Extended Position-Based Dynamics) framework
- Cosserat rod model for bending and twisting
- Separate stiffness controls (stretch, shear, bend, twist)
- Tip shaping for catheter/guidewire simulation
- BVH-accelerated mesh collision
- Friction models (Coulomb, viscous, static/dynamic)
"""

from .rod_data import (
    RodConfig,
    RodData,
    RodMaterialConfig,
    RodGeometryConfig,
    RodSolverConfig,
    RodTipConfig,
    FrictionConfig,
    CollisionMeshConfig,
)
from .rod_solver import RodSolver

__all__ = [
    "RodSolver",
    "RodConfig",
    "RodData",
    "RodMaterialConfig",
    "RodGeometryConfig",
    "RodSolverConfig",
    "RodTipConfig",
    "FrictionConfig",
    "CollisionMeshConfig",
]

