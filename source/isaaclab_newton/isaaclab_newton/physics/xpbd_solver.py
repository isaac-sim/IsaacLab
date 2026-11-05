# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD solver implementation."""

from newton.solvers import SolverXPBD

from .newton_solver import NewtonSolver


class XPBDSolver(NewtonSolver):
    """Newton's XPBD solver, which double-buffers state."""

    solver_class = SolverXPBD
    deterministic = True
