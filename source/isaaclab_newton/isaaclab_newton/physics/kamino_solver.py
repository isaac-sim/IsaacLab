# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino solver implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp
from newton import Model, State
from newton.solvers import SolverKamino

from .newton_solver import NewtonSolver

if TYPE_CHECKING:
    from .kamino_solver_cfg import _KaminoSolverCfgBase

logger = logging.getLogger(__name__)


class KaminoSolver(NewtonSolver):
    """Newton's Kamino solver, which resets its internals through forward kinematics."""

    cfg: _KaminoSolverCfgBase
    native: SolverKamino
    solver_class = SolverKamino

    @classmethod
    def coupling_error(cls, cfg: _KaminoSolverCfgBase, in_place: bool) -> str | None:
        return (
            "uses a Kamino solver config, whose FK/reset lifecycle cannot yet be preserved inside Newton's"
            " coupled-solver entry API"
        )

    def build_native(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverKamino:
        """Construct Kamino, capping contacts per world; the FK solver requires one articulation per environment.

        Unless :attr:`~_KaminoSolverCfgBase.use_fk_solver` chose explicitly, the FK solver is enabled for
        loop-closing articulations; :attr:`cfg` keeps the resolved copy.
        """
        cfg = self.cfg
        if cfg.use_fk_solver is None:
            start, end = model.articulation_start, model.articulation_end
            loop_closing = start is not None and end is not None and bool(end.shape[0])
            # Loop-closing joints are stored after an articulation's tree joints.
            loop_closing = loop_closing and bool((start.numpy()[1:] > end.numpy()).any())
            cfg = self.cfg = cfg.replace(use_fk_solver=loop_closing)
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
        return self.solver_class(model, cfg.to_solver_config())

    def init_state(self, state: State) -> None:
        """Initialize the output state's persistent Kamino buffers; FK initializes the input state."""
        self.native.reset(state, config=self.native.ResetConfig.preserve())

    def uses_collision_pipeline(self) -> bool:
        return not self.cfg.use_collision_detector

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Skip the generic reset; :meth:`eval_fk` performs the masked Kamino reset."""

    def eval_fk(self, state: State, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body state from joint coordinates and reset Kamino's internals for masked worlds.

        With :attr:`~_KaminoSolverCfgBase.use_fk_solver`, Kamino's loop-closure forward kinematics writes a consistent
        joint and body state. Otherwise Newton's articulated ``eval_fk`` runs over ``fk_mask``, and the caller is
        responsible for constraint-consistent joint values.
        """
        if self.cfg.use_fk_solver:
            self.native.reset(state, world_mask=world_mask, config=self.native.ResetConfig.from_joints())
            return
        super().eval_fk(state, world_mask, fk_mask)
        self.native.reset(state, world_mask=world_mask, config=self.native.ResetConfig.preserve())
