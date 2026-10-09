# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import ModelBuilder, State
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend


class NewtonVBDManager(NewtonManager):
    """:class:`NewtonManager` running the VBD solver, which double-buffers state."""

    solver_class = SolverVBD
    prepares_step = True

    @classmethod
    def prepare_solver_builder(cls, builder: ModelBuilder, solver_cfg: VBDSolverCfg) -> None:
        """Color the completed builder before allocating the model."""
        builder.color(balance_colors=False)

    @classmethod
    def supports_body_forces(cls, backend: NewtonBackend) -> bool:
        return not backend.cfg.solver_cfg.integrate_with_external_rigid_solver

    @classmethod
    def prepare_step(cls, backend: NewtonBackend, state: State) -> None:
        """Rebuild the particle BVH before each physics step."""
        if backend.model.particle_count > 0:
            backend.solver.rebuild_bvh(state)
