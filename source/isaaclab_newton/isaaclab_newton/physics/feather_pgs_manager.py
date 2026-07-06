# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS Newton manager."""

from __future__ import annotations

from newton import Model

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
    def _build_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> None:
        """Construct :class:`SolverFeatherPGS` and populate the base-class slots."""
        ignored = {"class_type", "solver_type"}
        kwargs = {key: value for key, value in solver_cfg.to_dict().items() if key not in ignored}
        NewtonManager._solver = SolverFeatherPGS(model, **kwargs)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
