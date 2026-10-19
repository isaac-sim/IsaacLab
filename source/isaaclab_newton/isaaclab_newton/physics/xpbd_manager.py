# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD Newton manager."""

from __future__ import annotations

import warp as wp
from newton import Model
from newton.solvers import SolverXPBD

from .newton_manager import NewtonManager
from .xpbd_manager_cfg import XPBDSolverCfg


class NewtonXPBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the XPBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def _create_solver(
        cls,
        model: Model,
        solver_cfg: XPBDSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverXPBD:
        """Construct the configured XPBD solver."""
        return SolverXPBD(model, **cls._filter_solver_kwargs(SolverXPBD, solver_cfg, deterministic_mode))

    def _build_solver(self, model: Model, solver_cfg: XPBDSolverCfg) -> None:
        """Construct :class:`SolverXPBD` and populate the solver slots.

        XPBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        self._solver = self._create_solver(model, solver_cfg, self._deterministic_mode)
        self._use_single_state = False
        self._needs_collision_pipeline = True
        self._supports_rigid_body_force_input = True
