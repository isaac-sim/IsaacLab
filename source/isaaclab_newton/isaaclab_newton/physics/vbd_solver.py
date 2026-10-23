# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD solver specification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import ModelBuilder, State, eval_ik
from newton.solvers import SolverVBD

from .solver_specs import SolverSpec, build_from_cfg

if TYPE_CHECKING:
    from .vbd_solver_cfg import VBDSolverCfg


def prepare_builder(builder: ModelBuilder, cfg: VBDSolverCfg) -> None:
    """Color the completed builder before allocating the model."""
    builder.color(include_bending=True, balance_colors=False)


def prepare_step(solver: SolverVBD, cfg: VBDSolverCfg, state: State) -> None:
    """Rebuild the particle BVH before each physics step."""
    if solver.model.particle_count > 0:
        solver.rebuild_bvh(state)


def finish_step(solver: SolverVBD, cfg: VBDSolverCfg, state: State) -> None:
    """Publish standalone VBD joint state without changing coupled-solver ownership."""
    model = solver.model
    if (
        cfg.update_joint_state
        and not cfg.integrate_with_external_rigid_solver
        and model.articulation_count > 0
        and model.joint_coord_count > 0
    ):
        eval_ik(model, state, state.joint_q, state.joint_qd)


VBD_SPEC = SolverSpec(
    prepare_builder=prepare_builder,
    build=build_from_cfg(SolverVBD),
    body_forces=lambda cfg: not cfg.integrate_with_external_rigid_solver,
    prepare_step=prepare_step,
    finish_step=finish_step,
)
"""Specification of the VBD solver, which double-buffers state."""
