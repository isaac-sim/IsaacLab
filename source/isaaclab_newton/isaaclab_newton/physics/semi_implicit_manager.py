# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semi-implicit Newton manager."""

from __future__ import annotations

from newton import Model
from newton.solvers import SolverSemiImplicit

from .newton_manager import NewtonManager
from .semi_implicit_manager_cfg import SemiImplicitSolverCfg


class NewtonSemiImplicitManager(NewtonManager):
    """:class:`NewtonManager` specialization for the semi-implicit solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: SemiImplicitSolverCfg) -> SolverSemiImplicit:
        """Construct the configured semi-implicit solver."""
        return SolverSemiImplicit(model, **cls._filter_solver_kwargs(SolverSemiImplicit, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: SemiImplicitSolverCfg) -> None:
        """Construct :class:`SolverSemiImplicit` and populate the base-class slots.

        The semi-implicit solver uses Newton's :class:`CollisionPipeline`,
        consumes rigid-body force input, and steps with separate input/output
        states, so the flags are fixed.
        """
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = True
