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

# Newton GPU XPBD rod solver (requires Newton build with SolverXPBDRod, e.g. PR #1981)
from .newton_xpbd_rod_wrapper import NewtonXPBDRodSolver, orientations_xyzw_along_polyline

# Self-contained XPBD rod solver — no external newton dependency
from .xpbd_rod_solver import XPBDRodSolver

# Vessel mesh collision extension — catheter-in-vessel containment + track guidance
from .xcath_rod_solver import (
    XCathRodSolver,
    COLLISION_PROJECTION_STAGE_POST,
    COLLISION_PROJECTION_STAGE_PRE,
    compute_smooth_vertex_normals,
    compute_signed_distances,
)

__all__ = [
    "RodSolver",
    "NewtonXPBDRodSolver",
    "XPBDRodSolver",
    "XCathRodSolver",
    "COLLISION_PROJECTION_STAGE_POST",
    "COLLISION_PROJECTION_STAGE_PRE",
    "compute_smooth_vertex_normals",
    "compute_signed_distances",
    "orientations_xyzw_along_polyline",
    "RodConfig",
    "RodData",
    "RodMaterialConfig",
    "RodGeometryConfig",
    "RodSolverConfig",
    "RodTipConfig",
    "FrictionConfig",
    "CollisionMeshConfig",
]

