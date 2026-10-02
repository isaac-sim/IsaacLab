# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

import warp as wp
from newton import Model, ModelBuilder, State
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .solver_binding import NewtonSolverBinding
from .vbd_manager_cfg import VBDSolverCfg


class VBDSolverBinding(NewtonSolverBinding):
    """Binding for the VBD solver, which double-buffers state and uses Newton's collision pipeline."""

    def __init__(self, model: Model, solver_cfg: VBDSolverCfg, deterministic_mode: wp.DeterministicMode):
        super().__init__(model, solver_cfg, deterministic_mode)
        self.supports_body_forces = not solver_cfg.integrate_with_external_rigid_solver

    @classmethod
    def prepare_builder(cls, builder: ModelBuilder) -> None:
        """Color the completed builder before allocating the model."""
        builder.color(balance_colors=False)

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: VBDSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **cls.filter_kwargs(SolverVBD, solver_cfg, deterministic_mode))

    def prepare_step(self, state: State) -> None:
        """Rebuild the particle BVH before each physics step."""
        if self.model.particle_count > 0:
            self.solver.rebuild_bvh(state)


class NewtonVBDManager(NewtonManager):
    """:class:`NewtonManager` running the VBD solver."""

    solver_binding = VBDSolverBinding
