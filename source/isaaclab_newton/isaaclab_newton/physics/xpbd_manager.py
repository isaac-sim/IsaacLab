# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD Newton manager."""

from __future__ import annotations

import inspect

from newton import Model
from newton.solvers import SolverXPBD

from .newton_manager import NewtonManager
from .xpbd_manager_cfg import XPBDSolverCfg


class NewtonXPBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the XPBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: XPBDSolverCfg) -> None:
        """Construct :class:`SolverXPBD` and populate the base-class slots.

        XPBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        valid = set(inspect.signature(SolverXPBD.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        NewtonManager._solver = SolverXPBD(model, **kwargs)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
