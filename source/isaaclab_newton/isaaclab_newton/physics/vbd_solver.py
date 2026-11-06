# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD solver implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton import CollisionPipeline, Model, ModelBuilder, State, eval_ik
from newton.solvers import SolverVBD

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .newton_manager_cfg import NewtonCollisionPipelineCfg
    from .vbd_solver_cfg import VBDSolverCfg


class VBDSolver(NewtonSolver):
    """Newton's Vertex Block Descent solver."""

    solver_class = SolverVBD

    @classmethod
    def prepare_builder(cls, cfg: VBDSolverCfg, builder: ModelBuilder) -> None:
        """Color the completed builder before allocating the model."""
        builder.color(include_bending=True, balance_colors=False)

    @classmethod
    def prepare_model(
        cls,
        cfg: VBDSolverCfg,
        model: Model,
        collision_cfg: NewtonCollisionPipelineCfg | None,
        mode: wp.DeterministicMode,
    ) -> CollisionPipeline | None:
        """Publish matching capacity before standalone rigid VBD allocates persistent contact history."""
        if not cfg.rigid_contact_history or model.body_count == 0 or cfg.integrate_with_external_rigid_solver:
            return None
        if collision_cfg is None or collision_cfg.contact_matching == "disabled":
            raise ValueError(
                "VBDSolverCfg.rigid_contact_history requires NewtonCfg.collision_cfg with "
                "contact_matching='latest' or 'sticky'."
            )
        args = collision_cfg.to_pipeline_args()
        args["deterministic"] = mode != wp.DeterministicMode.NOT_GUARANTEED
        return CollisionPipeline(model, **args)

    @classmethod
    def build(
        cls, cfg: VBDSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverVBD:
        """Construct VBD, leaving :attr:`~VBDSolverCfg.rigid_compliant_alm` to Newton's default when unset."""
        kwargs = solver_kwargs(cls.solver_class, cfg, mode)
        if cfg.rigid_compliant_alm is None:
            kwargs.pop("rigid_compliant_alm", None)
        return cls.solver_class(model, **kwargs)

    @classmethod
    def applies_body_forces(cls, cfg: VBDSolverCfg, solver: SolverVBD) -> bool:
        return not cfg.integrate_with_external_rigid_solver

    @classmethod
    def pre_step(cls, cfg: VBDSolverCfg, solver: SolverVBD, state: State) -> None:
        """Rebuild the particle BVH before each physics step."""
        if solver.model.particle_count > 0:
            solver.rebuild_bvh(state)

    @classmethod
    def post_step(cls, cfg: VBDSolverCfg, solver: SolverVBD, state: State) -> None:
        """Publish standalone VBD joint state without changing coupled-solver ownership."""
        model = solver.model
        if (
            cfg.update_joint_state
            and not cfg.integrate_with_external_rigid_solver
            and model.articulation_count > 0
            and model.joint_coord_count > 0
        ):
            eval_ik(model, state, state.joint_q, state.joint_qd)
