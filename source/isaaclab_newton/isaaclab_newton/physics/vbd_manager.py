# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from typing import ClassVar

import warp as wp
from newton import Model, ModelBuilder, eval_ik
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    _update_joint_state: ClassVar[bool] = True

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Color the completed builder before allocating the model."""
        builder.color(include_bending=True, balance_colors=False)

    @classmethod
    def _create_solver(
        cls,
        model: Model,
        solver_cfg: VBDSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg, deterministic_mode))

    def _build_solver(self, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct VBD and configure its base-manager state."""
        self._solver = self._create_solver(model, solver_cfg, self._deterministic_mode)
        self._use_single_state = False
        self._needs_collision_pipeline = True
        self._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver
        self._update_joint_state = solver_cfg.update_joint_state and not solver_cfg.integrate_with_external_rigid_solver

    def _run_solver_substeps(self, contacts) -> None:
        """Publish standalone VBD joint state without changing coupled-solver ownership."""
        super()._run_solver_substeps(contacts)
        if (
            isinstance(self._solver, SolverVBD)
            and self._update_joint_state
            and self.backend.model.articulation_count > 0
            and self.backend.model.joint_coord_count > 0
        ):
            state = self.backend.state_0
            eval_ik(self.backend.model, state, state.joint_q, state.joint_qd)

    def _simulate_physics_only(self) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if self.backend.model.particle_count > 0 and hasattr(self._solver, "rebuild_bvh"):
            self._solver.rebuild_bvh(self.backend.state_0)
        super()._simulate_physics_only()
