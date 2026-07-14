# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS Newton manager."""

from __future__ import annotations

from newton import Model, ModelBuilder

try:
    from newton.solvers import SolverFeatherPGS
except ImportError:
    from newton._src.solvers import SolverFeatherPGS

from .feather_pgs_manager_cfg import FeatherPGSSolverCfg
from .newton_manager import NewtonManager


class NewtonFeatherPGSManager(NewtonManager):
    """:class:`NewtonManager` specialization for the FeatherPGS solver.

    FeatherPGS uses Newton's :class:`CollisionPipeline` for contact handling and
    steps with separate input/output states.
    """

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the rigid-body custom attributes required by FeatherPGS."""
        if not builder.has_custom_attribute("rigid_body_max_linear_velocity"):
            SolverFeatherPGS.register_custom_attributes(builder)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> SolverFeatherPGS:
        """Construct the configured FeatherPGS solver without changing manager state."""
        return SolverFeatherPGS(model, **cls._filter_solver_kwargs(SolverFeatherPGS, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> None:
        """Construct :class:`SolverFeatherPGS` and populate the base-class slots."""
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
