# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from newton import Model, ModelBuilder, eval_ik
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Color the completed builder before allocating the model."""
        builder.color(include_bending=True, balance_colors=False)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct VBD and configure its base-manager state."""
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """Synchronize generalized joint state from VBD's maximal-coordinate body state."""
        super()._run_solver_substeps(contacts)
        if cls.backend.model.articulation_count > 0 and cls.backend.model.joint_coord_count > 0:
            state = cls.backend.state_0
            eval_ik(cls.backend.model, state, state.joint_q, state.joint_qd)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if cls.backend.model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls.backend.state_0)
        super()._simulate_physics_only()
