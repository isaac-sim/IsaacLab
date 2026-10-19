# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp
from newton import Model, State, eval_fk
from newton.solvers import SolverKamino

from .kamino_manager_cfg import _KaminoSolverCfgBase
from .newton_solver import NewtonSolver

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend

logger = logging.getLogger(__name__)


def _model_has_loop_closing_joints(model: Model) -> bool:
    """Return whether any articulation has loop-closing joints, stored after its tree joints."""
    if model.articulation_start is None or model.articulation_end is None:
        return False
    start, end = model.articulation_start.numpy(), model.articulation_end.numpy()
    return bool(end.shape[0] and (start[1:] > end).any())


class KaminoSolverAdapter(NewtonSolver):
    """:class:`NewtonSolver` adapter for the Kamino solver.

    Kamino treats body state as authoritative and double-buffers state. It uses Newton's collision pipeline unless
    ``use_collision_detector`` is ``True``, in which case Kamino's internal detector generates contacts.
    """

    builder_attribute_solvers = (SolverKamino,)

    def create_solver(
        self, model: Model, deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED
    ) -> SolverKamino:
        """Construct the configured Kamino solver; the FK solver requires one articulation per environment."""
        solver_cfg: _KaminoSolverCfgBase = self.cfg
        if solver_cfg.use_fk_solver is None:
            # Enable the FK solver for loop-closing articulations unless the user chose explicitly. The adapter keeps
            # the resolved copy, so eval_fk follows the same choice; the user's configuration stays untouched.
            solver_cfg = self.cfg = solver_cfg.replace(use_fk_solver=_model_has_loop_closing_joints(model))
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )
        if solver_cfg.use_fk_solver and model.articulation_count != model.world_count:
            raise RuntimeError(
                "The Kamino FK solver requires exactly one articulation per environment, but the model"
                f" has {model.articulation_count} articulations across {model.world_count} environments."
                " Multiple articulations per environment are not yet supported in Kamino's FK solver."
            )
        return SolverKamino(model, solver_cfg.to_solver_config())

    def uses_collision_pipeline(self, backend: NewtonBackend) -> bool:
        return not self.cfg.use_collision_detector

    def initialize_output_state(self, backend: NewtonBackend, state: State) -> None:
        """Initialize the output state's persistent Kamino buffers; FK initializes the input state."""
        backend.solver.reset(state, config=SolverKamino.ResetConfig.preserve())

    def reset_solver(self, backend: NewtonBackend, state: State, world_mask: wp.array) -> None:
        """Skip the generic reset; :meth:`eval_fk` performs the masked Kamino reset with an explicit configuration."""

    def eval_fk(
        self, backend: NewtonBackend, state: State, world_mask: wp.array | None, fk_mask: wp.array | None
    ) -> None:
        """Update body state from joint coordinates and reset Kamino's internals for masked worlds.

        With ``use_fk_solver``, :meth:`SolverKamino.reset` runs Kamino's loop-closure forward kinematics and writes a
        consistent joint and body state. Otherwise Newton's articulated ``eval_fk`` runs over ``fk_mask`` and the caller
        is responsible for constraint-consistent joint values.
        """
        solver = backend.solver
        if self.cfg.use_fk_solver:
            solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.from_joints())
            return
        eval_fk(backend.model, state.joint_q, state.joint_qd, state, fk_mask)
        solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.preserve())
