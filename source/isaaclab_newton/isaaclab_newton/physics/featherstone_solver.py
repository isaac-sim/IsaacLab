# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Featherstone solver specification."""

from __future__ import annotations

from types import MappingProxyType

from newton import ModelFlags
from newton.solvers import SolverFeatherstone

from .solver_specs import SolverSpec, build_from_cfg

FEATHERSTONE_SPEC = SolverSpec(
    build=build_from_cfg(SolverFeatherstone),
    deterministic=True,
    # SolverFeatherstone derives its inertia data from the model only when it is constructed.
    ignored_model_changes=MappingProxyType(
        {
            ModelFlags.BODY_INERTIAL_PROPERTIES: (
                "The Newton Featherstone solver does not apply mass, center of mass, or inertia changes made after"
                " the simulation starts; the simulation keeps the initial values."
            )
        }
    ),
)
"""Specification of the Featherstone solver, which double-buffers state."""
