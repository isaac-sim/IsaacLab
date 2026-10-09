# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD Newton manager."""

from __future__ import annotations

from newton.solvers import SolverXPBD

from .newton_manager import NewtonManager


class NewtonXPBDManager(NewtonManager):
    """:class:`NewtonManager` running the XPBD solver, which double-buffers state."""

    solver_class = SolverXPBD
    supports_heterogeneous_worlds = True
    supports_deterministic = True
