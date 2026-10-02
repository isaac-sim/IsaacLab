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
from .solver_binding import NewtonSolverBinding


class FeatherstoneSolverBinding(NewtonSolverBinding):
    """Binding for the Featherstone solver, which double-buffers state and uses Newton's collision pipeline."""

    def __init__(self, model: Model, solver_cfg: FeatherstoneSolverCfg, deterministic_mode: wp.DeterministicMode):
        super().__init__(model, solver_cfg, deterministic_mode)
        # SolverFeatherstone derives its inertia data from the model only when it is constructed.
        self.ignored_model_changes = {
            ModelFlags.BODY_INERTIAL_PROPERTIES: (
                "The Newton Featherstone solver does not apply mass, center of mass, or inertia changes made after"
                " the simulation starts; the simulation keeps the initial values."
            )
        }

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: FeatherstoneSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverFeatherstone:
        """Construct the configured Featherstone solver."""
        return SolverFeatherstone(model, **cls.filter_kwargs(SolverFeatherstone, solver_cfg, deterministic_mode))


class NewtonFeatherstoneManager(NewtonManager):
    """:class:`NewtonManager` running the Featherstone solver."""

    solver_binding = FeatherstoneSolverBinding
