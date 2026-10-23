# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino solver specification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp
from newton import Model, State, eval_fk
from newton.solvers import SolverKamino

from .kamino_solver_cfg import _KaminoSolverCfgBase
from .solver_specs import SolverSpec

if TYPE_CHECKING:
    from newton.solvers import SolverBase

logger = logging.getLogger(__name__)


def _model_has_loop_closing_joints(model: Model) -> bool:
    """Return whether any articulation has loop-closing joints, stored after its tree joints."""
    if model.articulation_start is None or model.articulation_end is None:
        return False
    start, end = model.articulation_start.numpy(), model.articulation_end.numpy()
    return bool(end.shape[0] and (start[1:] > end).any())


def resolve_cfg(model: Model, cfg: _KaminoSolverCfgBase) -> _KaminoSolverCfgBase:
    """Enable the FK solver for loop-closing articulations unless the configuration chose explicitly.

    The backend keeps the resolved copy, so :func:`eval_kamino_fk` follows the same choice; the user's configuration
    stays untouched.
    """
    if cfg.use_fk_solver is None:
        return cfg.replace(use_fk_solver=_model_has_loop_closing_joints(model))
    return cfg


def build(model: Model, cfg: _KaminoSolverCfgBase, deterministic_mode: wp.DeterministicMode) -> SolverKamino:
    """Construct the configured Kamino solver; the FK solver requires one articulation per environment."""
    cfg = resolve_cfg(model, cfg)
    if cfg.max_contacts_per_world is not None:
        model.rigid_contact_max = int(cfg.max_contacts_per_world) * model.world_count
        logger.info(
            "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
            model.rigid_contact_max,
            cfg.max_contacts_per_world,
            model.world_count,
        )
    if cfg.use_fk_solver and model.articulation_count != model.world_count:
        raise RuntimeError(
            "The Kamino FK solver requires exactly one articulation per environment, but the model"
            f" has {model.articulation_count} articulations across {model.world_count} environments."
            " Multiple articulations per environment are not yet supported in Kamino's FK solver."
        )
    return SolverKamino(model, cfg.to_solver_config())


def init_state(solver: SolverKamino, cfg: _KaminoSolverCfgBase, state: State) -> None:
    """Initialize the output state's persistent Kamino buffers; FK initializes the input state."""
    solver.reset(state, config=SolverKamino.ResetConfig.preserve())


def eval_kamino_fk(
    solver: SolverKamino,
    cfg: _KaminoSolverCfgBase,
    state: State,
    world_mask: wp.array | None,
    fk_mask: wp.array | None,
) -> None:
    """Update body state from joint coordinates and reset Kamino's internals for masked worlds.

    With ``use_fk_solver``, :meth:`SolverKamino.reset` runs Kamino's loop-closure forward kinematics and writes a
    consistent joint and body state. Otherwise Newton's articulated ``eval_fk`` runs over ``fk_mask`` and the caller
    is responsible for constraint-consistent joint values.
    """
    if cfg.use_fk_solver:
        solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.from_joints())
        return
    eval_fk(solver.model, state.joint_q, state.joint_qd, state, fk_mask)
    solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.preserve())


def _reset(solver: SolverBase, cfg: _KaminoSolverCfgBase, state: State, world_mask: wp.array) -> None:
    """Skip the generic reset; :func:`eval_kamino_fk` performs the masked Kamino reset with an explicit
    configuration."""


KAMINO_SPEC = SolverSpec(
    attribute_solvers=lambda cfg: (SolverKamino,),
    resolve_cfg=resolve_cfg,
    build=build,
    init_state=init_state,
    coupling_error=lambda cfg, in_place: (
        "uses a Kamino solver config, whose FK/reset lifecycle cannot yet be preserved inside Newton's"
        " coupled-solver entry API"
    ),
    external_contacts=lambda solver, cfg: not cfg.use_collision_detector,
    reset=_reset,
    eval_fk=eval_kamino_fk,
)
"""Specification of the Kamino solver.

Kamino treats body state as authoritative and double-buffers state. It uses Newton's collision pipeline unless
``use_collision_detector`` is ``True``, in which case Kamino's internal detector generates contacts.
"""
