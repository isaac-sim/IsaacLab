# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD solver specification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton import ModelBuilder, State, eval_ik
from newton.solvers import SolverVBD

from .solver_specs import SolverSpec, solver_kwargs

if TYPE_CHECKING:
    from .vbd_solver_cfg import VBDSolverCfg


def prepare_builder(builder: ModelBuilder, cfg: VBDSolverCfg) -> None:
    """Color the completed builder before allocating the model."""
    builder.color(include_bending=True, balance_colors=False)


def vbd_kwargs(cfg: VBDSolverCfg, deterministic_mode: wp.DeterministicMode) -> dict:
    """Return :class:`SolverVBD` arguments, leaving ``rigid_compliant_alm`` to Newton's default when unset."""
    kwargs = solver_kwargs(SolverVBD, cfg, deterministic_mode)
    if cfg.rigid_compliant_alm is None:
        del kwargs["rigid_compliant_alm"]
    return kwargs


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
    build=lambda model, cfg, deterministic_mode: SolverVBD(model, **vbd_kwargs(cfg, deterministic_mode)),
    body_forces=lambda cfg: not cfg.integrate_with_external_rigid_solver,
    prepare_step=prepare_step,
    finish_step=finish_step,
)
"""Specification of the VBD solver, which double-buffers state."""
