# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class FeatherstoneSolverCfg(NewtonSolverCfg):
    """A semi-implicit integrator using symplectic Euler.

    It operates on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it is not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
    """

    class_type: type[NewtonManager] | str = "{DIR}.featherstone_manager:NewtonFeatherstoneManager"
    """Manager class for the Featherstone solver."""

    solver_type: str = "featherstone"
    """Solver type. Can be "featherstone"."""

    angular_damping: float = 0.05
    """Angular damping parameter for rigid contact simulation."""

    update_mass_matrix_interval: int = 1
    """Frequency (in simulation steps) at which to update the mass matrix."""

    friction_smoothing: float = 1.0
    """Friction smoothing parameter."""

    use_tile_gemm: bool = False
    """Whether to use tile-based GEMM for the mass matrix."""

    fuse_cholesky: bool = True
    """Whether to fuse the Cholesky decomposition."""
