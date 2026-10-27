# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD solver implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton import Model, ModelBuilder, State, eval_ik
from newton.solvers import SolverVBD

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .vbd_solver_cfg import VBDSolverCfg


class VBDSolver(NewtonSolver):
    """Newton's Vertex Block Descent solver."""

    cfg: VBDSolverCfg
    solver_class = SolverVBD

    @classmethod
    def prepare_builder(cls, cfg: VBDSolverCfg, builder: ModelBuilder) -> None:
        """Color the completed builder before allocating the model."""
        builder.color(include_bending=True, balance_colors=False)

    def build_native(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverVBD:
        """Construct VBD, leaving :attr:`~VBDSolverCfg.rigid_compliant_alm` to Newton's default when unset."""
        kwargs = solver_kwargs(self.solver_class, self.cfg, mode)
        if self.cfg.rigid_compliant_alm is None:
            kwargs.pop("rigid_compliant_alm", None)
        return self.solver_class(model, **kwargs)

    def applies_body_forces(self) -> bool:
        return not self.cfg.integrate_with_external_rigid_solver

    def pre_step(self, state: State) -> None:
        """Rebuild the particle BVH before each physics step."""
        if self.native.model.particle_count > 0:
            self.native.rebuild_bvh(state)

    def post_step(self, state: State) -> None:
        """Publish standalone VBD joint state without changing coupled-solver ownership."""
        model = self.native.model
        if (
            self.cfg.update_joint_state
            and not self.cfg.integrate_with_external_rigid_solver
            and model.articulation_count > 0
            and model.joint_coord_count > 0
        ):
            eval_ik(model, state, state.joint_q, state.joint_qd)
