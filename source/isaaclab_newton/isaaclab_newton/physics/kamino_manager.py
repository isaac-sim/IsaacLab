# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, State, eval_fk
from newton.solvers import SolverKamino

from .kamino_manager_cfg import _KaminoSolverCfgBase
from .newton_manager import NewtonManager
from .solver_binding import NewtonSolverBinding

logger = logging.getLogger(__name__)


def _model_has_loop_closing_joints(model: Model) -> bool:
    """Return whether ``model`` contains converted loop-closing articulation joints.

    Newton stores regular tree joints in ``[articulation_start[i], articulation_end[i])`` and
    loop-closing joints in ``[articulation_end[i], articulation_start[i + 1])``. Loop closures
    are present when the next articulation sentinel exceeds the tree joint end for any
    articulation.

    Args:
        model: Finalized Newton model to inspect.

    Returns:
        ``True`` if at least one articulation has loop-closing joints.
    """
    articulation_start = model.articulation_start
    articulation_end = model.articulation_end
    if articulation_start is None or articulation_end is None:
        return False
    articulation_start_np = articulation_start.numpy()
    articulation_end_np = articulation_end.numpy()
    if articulation_end_np.shape[0] == 0:
        return False
    return bool((articulation_start_np[1:] > articulation_end_np).any())


class KaminoSolverBinding(NewtonSolverBinding):
    """Binding for the Kamino solver.

    Kamino treats body state as authoritative and double-buffers state. It uses Newton's collision pipeline unless
    ``use_collision_detector`` is ``True``, in which case Kamino's internal detector generates contacts.
    """

    builder_attribute_solvers = (SolverKamino,)

    solver: SolverKamino

    def __init__(self, model: Model, solver_cfg: _KaminoSolverCfgBase, deterministic_mode: wp.DeterministicMode):
        """Construct the solver.

        Raises:
            RuntimeError: If the FK solver is enabled with more than one articulation per environment.
        """
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )
        # Enable the FK solver for loop-closing articulations unless the user chose explicitly.
        if solver_cfg.use_fk_solver is None:
            solver_cfg.use_fk_solver = _model_has_loop_closing_joints(model)
        if solver_cfg.use_fk_solver and model.articulation_count != model.world_count:
            raise RuntimeError(
                "The Kamino FK solver requires exactly one articulation per environment, but the model"
                f" has {model.articulation_count} articulations across {model.world_count} environments."
                " Multiple articulations per environment are not yet supported in Kamino's FK solver."
            )
        super().__init__(model, solver_cfg, deterministic_mode)
        self.needs_collision_pipeline = not solver_cfg.use_collision_detector

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: _KaminoSolverCfgBase,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverKamino:
        """Construct the configured Kamino solver."""
        return SolverKamino(model, solver_cfg.to_solver_config())

    def initialize_output_state(self, state: State) -> None:
        """Initialize the output state's persistent Kamino buffers; FK initializes the input state."""
        self.solver.reset(state, config=SolverKamino.ResetConfig.preserve())

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Skip the generic reset; :meth:`eval_fk` performs the masked Kamino reset with an explicit configuration."""

    def eval_fk(self, state: State, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body state from joint coordinates and reset Kamino's internals for masked worlds.

        With ``use_fk_solver``, :meth:`SolverKamino.reset` runs Kamino's loop-closure forward kinematics and writes a
        consistent joint and body state. Otherwise Newton's articulated ``eval_fk`` runs over ``fk_mask`` and the caller
        is responsible for constraint-consistent joint values.
        """
        if self.cfg.use_fk_solver:
            self.solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.from_joints())
            return
        eval_fk(self.model, state.joint_q, state.joint_qd, state, fk_mask)
        self.solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.preserve())


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` running the Kamino solver."""

    solver_binding = KaminoSolverBinding
