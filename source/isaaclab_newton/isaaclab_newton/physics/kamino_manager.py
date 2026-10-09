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
from .newton_manager import NewtonManager

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend

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


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` running the Kamino solver.

    Kamino treats body state as authoritative and double-buffers state. It uses Newton's collision pipeline unless
    ``use_collision_detector`` is ``True``, in which case Kamino's internal detector generates contacts.
    """

    builder_attribute_solvers = (SolverKamino,)
    supports_heterogeneous_worlds = True

    @classmethod
    def create_solver(
        cls,
        model: Model,
        solver_cfg: _KaminoSolverCfgBase,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverKamino:
        """Construct the configured Kamino solver.

        Raises:
            RuntimeError: If the FK solver is enabled with more than one articulation per environment.
        """
        if solver_cfg.use_fk_solver is None:
            # Enable the FK solver for loop-closing articulations unless the user chose explicitly.
            solver_cfg = solver_cfg.replace(use_fk_solver=_model_has_loop_closing_joints(model))
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

    @classmethod
    def validate_cfg(cls, backend: NewtonBackend) -> None:
        """Resolve the automatic FK-solver choice into the backend's configuration, leaving the user's untouched."""
        super().validate_cfg(backend)
        solver_cfg = backend.cfg.solver_cfg
        if solver_cfg.use_fk_solver is None:
            use_fk_solver = _model_has_loop_closing_joints(backend.model)
            backend.cfg = backend.cfg.replace(solver_cfg=solver_cfg.replace(use_fk_solver=use_fk_solver))

    @classmethod
    def uses_collision_pipeline(cls, backend: NewtonBackend) -> bool:
        return not backend.cfg.solver_cfg.use_collision_detector

    @classmethod
    def initialize_output_state(cls, backend: NewtonBackend, state: State) -> None:
        """Initialize the output state's persistent Kamino buffers; FK initializes the input state."""
        backend.solver.reset(state, config=SolverKamino.ResetConfig.preserve())

    @classmethod
    def reset_solver(cls, backend: NewtonBackend, state: State, world_mask: wp.array) -> None:
        """Skip the generic reset; :meth:`eval_fk` performs the masked Kamino reset with an explicit configuration."""

    @classmethod
    def eval_fk(
        cls, backend: NewtonBackend, state: State, world_mask: wp.array | None, fk_mask: wp.array | None
    ) -> None:
        """Update body state from joint coordinates and reset Kamino's internals for masked worlds.

        With ``use_fk_solver``, :meth:`SolverKamino.reset` runs Kamino's loop-closure forward kinematics and writes a
        consistent joint and body state. Otherwise Newton's articulated ``eval_fk`` runs over ``fk_mask`` and the caller
        is responsible for constraint-consistent joint values.
        """
        solver = backend.solver
        if backend.cfg.solver_cfg.use_fk_solver:
            solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.from_joints())
            return
        eval_fk(backend.model, state.joint_q, state.joint_qd, state, fk_mask)
        solver.reset(state, world_mask=world_mask, config=SolverKamino.ResetConfig.preserve())
