# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the deformable lift tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
