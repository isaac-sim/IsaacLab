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
    """:class:`NewtonManager` specialization for the Featherstone solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def _create_solver(
        cls,
        model: Model,
        solver_cfg: FeatherstoneSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverFeatherstone:
        """Construct the configured Featherstone solver."""
        return SolverFeatherstone(
            model, **cls._filter_solver_kwargs(SolverFeatherstone, solver_cfg, deterministic_mode)
        )

    def _build_solver(self, model: Model, solver_cfg: FeatherstoneSolverCfg) -> None:
        """Construct :class:`SolverFeatherstone` and populate the solver slots.

        Featherstone always uses Newton's :class:`CollisionPipeline` and steps
        with separate input/output states, so the flags are fixed.
        """
        self._solver = self._create_solver(model, solver_cfg, self._deterministic_mode)
        self._use_single_state = False
        self._needs_collision_pipeline = True
        self._supports_rigid_body_force_input = True
        # SolverFeatherstone derives its inertia data from the model only when it is constructed
        self._ignored_model_changes = {
            ModelFlags.BODY_INERTIAL_PROPERTIES: (
                "The Newton Featherstone solver does not apply mass, center of mass, or inertia changes made after"
                " the simulation starts; the simulation keeps the initial values."
            )
        }
