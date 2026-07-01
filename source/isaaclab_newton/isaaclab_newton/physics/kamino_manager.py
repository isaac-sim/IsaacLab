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
    def _eval_fk_impl(cls, fk_mask: wp.array | None) -> None:
        """Update body states from joint coordinates.

        For the Kamino (maximal-coordinate) solver, body poses/velocities are the authoritative
        simulation state. When :attr:`KaminoSolverCfg.use_fk_solver` is enabled, this calls
        :meth:`SolverKamino.reset`, which runs Kamino's loop-closure forward kinematics: it reads
        body poses/velocities from the joint coordinates (including the base body's pose/twist)
        and writes back a consistent full joint and body state.

        When ``use_fk_solver`` is disabled, falls back to Newton's articulated ``eval_fk`` over
        ``fk_mask``; the caller is then responsible for writing constraint-consistent joint values.

        Args:
            fk_mask: Per-articulation mask of articulations to update (``None`` means all).
        """
        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._solver.reset(
                cls._state_0,
                world_mask=fk_mask,
                config=SolverKamino.ResetConfig.from_joints(),
            )
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, fk_mask)

            # Reset solver internals without performing Kamino's FK.
            cls._solver.reset(
                cls._state_0,
                world_mask=fk_mask,
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
        step. The shared :attr:`NewtonManager._fk_reset_mask` restricts both that pre-step
        reconcile and :meth:`NewtonManager.forward` to reset worlds, so non-reset worlds keep
        their live, authoritative body state through Kamino's :meth:`_eval_fk_impl` overwrite.

        Raises:
            RuntimeError: If the model has more than one articulation per environment. The Kamino
                interface in IsaacLab currently only supports one articulation per environment.
        """
        if model.articulation_count != model.world_count:
            raise RuntimeError(
                "The Kamino manager requires exactly one articulation per environment, but the model"
                f" has {model.articulation_count} articulations across {model.world_count} environments."
                " Multiple articulations per environment are not yet supported in IsaacLab's Kamino manager."
            )
        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector
        NewtonManager._needs_fk_before_step = True
