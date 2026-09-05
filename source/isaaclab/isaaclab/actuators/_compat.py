# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated actuator-configuration compatibility helpers.

Everything in this module supports configuration fields and constructor
arguments that are deprecated in Isaac Lab 3.x and scheduled for removal in
4.0. Delete this module together with the aliases.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.string import _resolve_matching_values_dense

from .actuator_base_cfg import _is_implicit_actuator_cfg

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg


def _limits_equal(first: torch.Tensor | float, second: torch.Tensor | float) -> bool:
    """Return whether two constructor limit arguments are equivalent."""
    if isinstance(first, torch.Tensor):
        if isinstance(second, torch.Tensor):
            return first.shape == second.shape and torch.equal(first, second)
        return bool(torch.all(first == float(second)).item())
    if isinstance(second, torch.Tensor):
        return bool(torch.all(second == float(first)).item())
    return float(first) == float(second)


def _resolve_limit_aliases(
    actuator_name: str,
    cfg: ActuatorBaseCfg,
    joint_names: list[str],
    *,
    warn_deprecated: bool = True,
) -> None:
    """Normalize deprecated effort- and velocity-limit aliases on an actuator configuration.

    Writes the configuration in place: canonical fields absorb the alias values after
    equivalence validation and the alias fields are cleared. The collection resolves on
    its internal configuration copies; directly constructed actuators resolve on the
    configuration object they were given.
    """
    implicit = _is_implicit_actuator_cfg(cfg)
    if implicit and cfg.joint_effort_limit is None and cfg.effort_limit_sim is None and cfg.effort_limit is not None:
        # Deprecated implicit behavior: without a separate solver clamp, the rated
        # effort limit also reaches the solver.
        cfg.joint_effort_limit = cfg.effort_limit
    if implicit and cfg.actuator_velocity_limit is None and cfg.velocity_limit is None and cfg.velocity_limit_sim is not None:
        # Deprecated implicit behavior: the solver clamp doubles as the soft joint
        # velocity limit so the data buffers stay meaningful.
        cfg.actuator_velocity_limit = cfg.velocity_limit_sim

    for new_name, old_name in (
        ("joint_effort_limit", "effort_limit_sim"),
        ("actuator_effort_limit", "effort_limit"),
        ("joint_velocity_limit", "velocity_limit_sim"),
        ("actuator_velocity_limit", "velocity_limit"),
    ):
        alias_value = getattr(cfg, old_name)
        if alias_value is None:
            continue
        if warn_deprecated:
            warnings.warn(
                f"Actuator group '{actuator_name}' uses deprecated '{old_name}'. Use "
                f"'{new_name}' instead; '{old_name}' will be removed in 4.0.",
                DeprecationWarning,
                stacklevel=3,
            )
        new_value = getattr(cfg, new_name)
        if new_value is None:
            setattr(cfg, new_name, alias_value)
        elif _resolve_matching_values_dense(new_value, joint_names) != _resolve_matching_values_dense(
            alias_value, joint_names
        ):
            raise ValueError(
                f"Actuator group '{actuator_name}' has conflicting '{new_name}' and "
                f"deprecated '{old_name}' values."
            )
        setattr(cfg, old_name, None)
