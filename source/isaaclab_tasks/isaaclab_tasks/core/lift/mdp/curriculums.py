# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the lift environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def difficulty_interpolate_float(
    env: ManagerBasedRLEnv,
    _env_ids: Sequence[int],
    _data: float,
    initial_value: float,
    final_value: float,
    difficulty_term_str: str,
) -> float:
    """Interpolate a scalar continuously with an ADR term's success-driven difficulty."""
    difficulty_term = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    fraction = min(max(difficulty_term.difficulty_frac, 0.0), 1.0)
    return initial_value + fraction * (final_value - initial_value)


def gravity_range_linear(
    env: ManagerBasedRLEnv,
    _env_ids: Sequence[int],
    event_name: str,
    start_gravity_z: float,
    end_gravity_z: float,
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Linearly ramp an event's deterministic vertical gravity [m/s^2]."""
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step.")

    alpha = (env.common_step_counter - start_step) / (end_step - start_step)
    alpha = min(max(alpha, 0.0), 1.0)
    gravity_z = start_gravity_z + alpha * (end_gravity_z - start_gravity_z)
    gravity = [0.0, 0.0, gravity_z]
    event_cfg = env.event_manager.get_term_cfg(event_name)
    event_cfg.params["gravity_distribution_params"] = (gravity, gravity.copy())
    env.event_manager.set_term_cfg(event_name, event_cfg)
    return {"gravity_z": gravity_z}
