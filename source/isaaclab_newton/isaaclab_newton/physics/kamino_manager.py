# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, eval_fk
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

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
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

    # Annotate the concrete solver type.
    _solver: SolverKamino

    @classmethod
    def _get_kamino_solver_cfg(cls) -> KaminoSolverCfg:
        cfg = PhysicsManager._cfg
        if cfg is None:
            raise RuntimeError("Physics manager is not initialized.")
        solver_cfg = getattr(cfg, "solver_cfg", None)
        if not isinstance(solver_cfg, KaminoSolverCfg):
            raise TypeError(f"Expected KaminoSolverCfg, got {type(solver_cfg).__name__}.")
        return solver_cfg

    @classmethod
    def _eval_fk_impl(cls, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body states from joint coordinates.

        For the Kamino (maximal-coordinate) solver, body poses/velocities are the authoritative
        simulation state. When :attr:`KaminoSolverCfg.use_fk_solver` is enabled, this calls
        :meth:`SolverKamino.reset`, which runs Kamino's loop-closure forward kinematics: it reads
        body poses/velocities from the joint coordinates (including the base body's pose/twist)
        and writes back a consistent full joint and body state.

        When ``use_fk_solver`` is disabled, falls back to Newton's articulated ``eval_fk`` over
        ``fk_mask``; the caller is then responsible for writing constraint-consistent joint values.

        Args:
            world_reset_mask: Per-world mask passed to :meth:`SolverKamino.reset` (``None`` means all).
            fk_mask: Per-articulation mask of articulations to update (``None`` means all).
        """
        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._solver.reset(
                cls._state_0,
                world_mask=world_reset_mask,
                config=SolverKamino.ResetConfig.from_joints(),
            )
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, fk_mask)

            # Reset solver internals without performing Kamino's FK.
            cls._solver.reset(
                cls._state_0,
                world_mask=world_reset_mask,
                config=SolverKamino.ResetConfig.preserve(),
            )

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).

        Sets :attr:`NewtonManager._needs_fk_before_step` because Kamino treats body state as
        authoritative: reset worlds (written via joint coordinates) must be reconciled before each
        step. The shared :attr:`NewtonManager._world_reset_mask` and
        :attr:`NewtonManager._fk_reset_mask` restrict that pre-step reconcile and
        :meth:`NewtonManager.forward` to reset worlds, so non-reset worlds keep their live,
        authoritative body state through Kamino's :meth:`_eval_fk_impl` overwrite.

        Raises:
            RuntimeError: If the model has more than one articulation per environment. The Kamino
                interface in IsaacLab currently only supports one articulation per environment.
        """

        # Set the max contacts per world if specified.
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )

        # Set the use_fk_solver flag based on the model's articulation structure if not specified by user.
        if solver_cfg.use_fk_solver is None:
            solver_cfg.use_fk_solver = _model_has_loop_closing_joints(model)

        if solver_cfg.use_fk_solver and model.articulation_count != model.world_count:
            raise RuntimeError(
                "The Kamino FK solver requires exactly one articulation per environment, but the model"
                f" has {model.articulation_count} articulations across {model.world_count} environments."
                " Multiple articulations per environment are not yet supported in Kamino's FK solver."
            )

        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector
        NewtonManager._needs_fk_before_step = True
