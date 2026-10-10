# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Featherstone Newton manager."""

from __future__ import annotations

from newton import ModelFlags
from newton.solvers import SolverFeatherstone

from .newton_solver import NewtonSolver


class FeatherstoneSolverAdapter(NewtonSolver):
    """:class:`NewtonSolver` adapter for the Featherstone solver, which double-buffers state."""

    solver_class = SolverFeatherstone
    supports_heterogeneous_worlds = True
    supports_deterministic = True
    # SolverFeatherstone derives its inertia data from the model only when it is constructed.
    ignored_model_changes = {
        ModelFlags.BODY_INERTIAL_PROPERTIES: (
            "The Newton Featherstone solver does not apply mass, center of mass, or inertia changes made after"
            " the simulation starts; the simulation keeps the initial values."
        )
    }
