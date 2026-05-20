# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

from newton import Model
from newton.solvers import SolverImplicitMPM

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager


class NewtonMPMManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton's implicit MPM solver.

    MPM handles particle/collider interaction internally and does not consume
    Newton's rigid-body collision pipeline.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> None:
        """Construct :class:`SolverImplicitMPM` and populate the base-class slots.

        Args:
            model: Finalized Newton model the solver should run on.
            solver_cfg: Implicit MPM solver configuration.
        """
        if model.particle_count == 0:
            raise ValueError(
                "Newton implicit MPM requires at least one particle. Add particles to the Newton builder before "
                "starting the simulation."
            )

        NewtonManager._solver = SolverImplicitMPM(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
