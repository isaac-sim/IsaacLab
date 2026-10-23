# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XPBD solver specification."""

from __future__ import annotations

from newton.solvers import SolverXPBD

from .solver_specs import SolverSpec, build_from_cfg

XPBD_SPEC = SolverSpec(build=build_from_cfg(SolverXPBD), deterministic=True)
"""Specification of the XPBD solver, which double-buffers state."""
