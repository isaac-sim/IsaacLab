# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Featherstone Newton manager."""

from __future__ import annotations

import warp as wp
from newton import Model, ModelFlags
from newton.solvers import SolverFeatherstone

from .featherstone_manager_cfg import FeatherstoneSolverCfg
from .newton_manager import NewtonManager


class NewtonFeatherstoneManager(NewtonManager):
    """:class:`NewtonManager` running the Featherstone solver, which double-buffers state."""

    supports_deterministic = True
    # SolverFeatherstone derives its inertia data from the model only when it is constructed.
    ignored_model_changes = {
        ModelFlags.BODY_INERTIAL_PROPERTIES: (
            "The Newton Featherstone solver does not apply mass, center of mass, or inertia changes made after"
            " the simulation starts; the simulation keeps the initial values."
        )
    }

    @classmethod
    def create_solver(
        cls,
        model: Model,
        solver_cfg: FeatherstoneSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverFeatherstone:
        """Construct the configured Featherstone solver."""
        return SolverFeatherstone(model, **cls.solver_kwargs(SolverFeatherstone, solver_cfg, deterministic_mode))
