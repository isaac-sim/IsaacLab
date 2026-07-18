# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public API gates for :class:`FeatherPGSSolverCfg`."""

from __future__ import annotations

import inspect
from typing import Literal, get_args, get_origin, get_type_hints

try:
    from newton.solvers import SolverFeatherPGS
except ImportError:
    from newton._src.solvers import SolverFeatherPGS

import isaaclab_newton.physics.feather_pgs_manager_cfg as feather_pgs_cfg_module
from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonManager

_STRING_OPTION_DOMAINS = {
    "pgs_velocity_drive_mode": ("active", "freeze"),
    "pgs_mode": ("dense", "split", "matrix_free"),
    "pgs_schedule": ("interleaved", "contact_then_internal", "physx_grasp"),
    "friction_mode": ("current", "bisection", "bisection_desaxce", "coulomb_newton"),
    "cholesky_kernel": ("tiled", "loop", "auto"),
    "trisolve_kernel": ("tiled", "loop", "auto"),
    "hinv_jt_kernel": ("tiled", "par_row", "auto"),
    "delassus_kernel": ("tiled", "par_row_col", "auto"),
    "pgs_kernel": ("loop", "tiled_row", "tiled_contact", "streaming"),
    "drive_mode": ("augmented", "physx_pgs"),
    "effort_limit_mode": ("actuator",),
}


def test_feather_pgs_cfg_preserves_closed_string_option_domains():
    """Constrained public solver options must retain precise ``Literal`` annotations."""
    hints = get_type_hints(
        FeatherPGSSolverCfg,
        globalns={**vars(feather_pgs_cfg_module), "NewtonManager": NewtonManager},
    )

    for field_name, expected_values in _STRING_OPTION_DOMAINS.items():
        assert get_origin(hints[field_name]) is Literal
        assert get_args(hints[field_name]) == expected_values


def test_feather_pgs_cfg_fields_and_defaults_match_installed_solver():
    """Every public FeatherPGS field must exist upstream and preserve its installed default."""
    cfg = FeatherPGSSolverCfg()
    parameters = inspect.signature(SolverFeatherPGS.__init__).parameters

    for field_name in FeatherPGSSolverCfg.__annotations__:
        if field_name in {"class_type", "solver_type"}:
            continue
        assert field_name in parameters
        assert getattr(cfg, field_name) == parameters[field_name].default
