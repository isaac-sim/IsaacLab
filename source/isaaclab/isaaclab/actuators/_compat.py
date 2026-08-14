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

from .actuator_base_cfg import _is_implicit_actuator_cfg, _resolve_limit_values

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg


def _effort_limits_equal(first: torch.Tensor | float, second: torch.Tensor | float) -> bool:
    """Return whether two constructor effort-limit arguments are equivalent."""
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

    The caller owns the configuration copy because this function writes canonical
    values after validating equivalent scalar or regex configurations.
    """
    implicit = _is_implicit_actuator_cfg(cfg)
    if implicit and cfg.actuator_effort_limit is not None:
        raise ValueError(
            f"Implicit actuator group '{actuator_name}' cannot set 'actuator_effort_limit'. "
            "Use 'joint_effort_limit' for the solver limit."
        )
    if implicit and cfg.velocity_limit is None and cfg.velocity_limit_sim is not None:
        # Deprecated implicit behavior: the solver clamp doubles as the soft joint
        # velocity limit so the data buffers stay meaningful.
        cfg.velocity_limit = cfg.velocity_limit_sim

    for canonical_name, alias_name in (
        ("joint_effort_limit", "effort_limit_sim"),
        ("joint_effort_limit" if implicit else "actuator_effort_limit", "effort_limit"),
        ("joint_velocity_limit", "velocity_limit_sim"),
    ):
        alias_value = getattr(cfg, alias_name)
        if alias_value is None:
            continue
        if warn_deprecated:
            warnings.warn(
                f"Actuator group '{actuator_name}' uses deprecated '{alias_name}'. Use "
                f"'{canonical_name}' instead; '{alias_name}' will be removed in 4.0.",
                DeprecationWarning,
                stacklevel=3,
            )
        canonical_value = getattr(cfg, canonical_name)
        if canonical_value is None:
            setattr(cfg, canonical_name, alias_value)
        elif _resolve_limit_values(canonical_value, joint_names) != _resolve_limit_values(alias_value, joint_names):
            raise ValueError(
                f"Actuator group '{actuator_name}' has conflicting '{canonical_name}' and "
                f"deprecated '{alias_name}' values."
            )
        setattr(cfg, alias_name, None)
