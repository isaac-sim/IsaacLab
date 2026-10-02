# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD Newton manager."""

from __future__ import annotations

import warp as wp
from newton import Model
from newton.solvers import SolverXPBD

from .newton_manager import NewtonManager
from .solver_binding import NewtonSolverBinding
from .xpbd_manager_cfg import XPBDSolverCfg


class XPBDSolverBinding(NewtonSolverBinding):
    """Binding for the XPBD solver, which double-buffers state and uses Newton's collision pipeline."""

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: XPBDSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverXPBD:
        """Construct the configured XPBD solver."""
        return SolverXPBD(model, **cls.filter_kwargs(SolverXPBD, solver_cfg, deterministic_mode))


class NewtonXPBDManager(NewtonManager):
    """:class:`NewtonManager` running the XPBD solver."""

    solver_binding = XPBDSolverBinding
