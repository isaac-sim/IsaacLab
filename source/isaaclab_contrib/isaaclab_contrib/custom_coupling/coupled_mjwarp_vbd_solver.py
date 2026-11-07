# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated MJWarp and VBD configuration adapter for the generic proxy coupler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonSolver
from isaaclab_newton.physics.coupling import CouplerSolver
from newton import Model, ModelBuilder
from newton.solvers import SolverBase

if TYPE_CHECKING:
    from .coupled_mjwarp_vbd_solver_cfg import CoupledMJWarpVBDSolverCfg


class CoupledMJWarpVBDSolver(CouplerSolver):
    """Deprecated adapter to :class:`~isaaclab_newton.physics.coupling.CouplerSolver`.

    Existing configurations use proxy feedback, with different numerical behavior from the former shared-model
    penalty coupling. The driver exposes the translated proxy configuration through :attr:`cfg`.
    New configurations should select :class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg`.
    """

    @classmethod
    def solver_classes(cls, cfg: CoupledMJWarpVBDSolverCfg) -> tuple[type[SolverBase], ...]:
        return CouplerSolver.solver_classes(cfg._to_proxy_cfg())

    @classmethod
    def prepare_builder(cls, cfg: CoupledMJWarpVBDSolverCfg, builder: ModelBuilder) -> None:
        CouplerSolver.prepare_builder(cfg._to_proxy_cfg(), builder)

    def __init__(
        self, cfg: CoupledMJWarpVBDSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> None:
        super().__init__(cfg._to_proxy_cfg(), model, mode, env_template)

    @property
    def rigid(self) -> NewtonSolver:
        """Rigid solver driver, retained for compatibility; use :attr:`entries` in new code."""
        return self.entries["rigid"]

    @property
    def soft(self) -> NewtonSolver:
        """Deformable solver driver, retained for compatibility; use :attr:`entries` in new code."""
        return self.entries["soft"]
