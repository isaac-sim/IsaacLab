# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployable head-sensor inputs for the MicroDuck command policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tof_ranges_and_validity(env):
    """Flatten metre ranges and a separate validity mask; missing returns are not free space."""
    sensor = env.scene["head_tof"]
    ranges = sensor.observation
    valid = (
        torch.isfinite(ranges)
        & (ranges >= sensor.cfg.randomization.minimum_distance)
        & (ranges <= sensor.cfg.max_distance)
    )
    ranges = torch.where(valid, ranges, 0.0)
    return torch.cat((ranges.flatten(1), valid.float().flatten(1)), dim=-1)


def head_sensor_age(env):
    """ToF frame age in seconds, including transport delay and dropped frames."""
    return env.scene["head_tof"].sample_age.unsqueeze(-1)


def applied_velocity_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The three commands actually delivered to the frozen walker."""
    return env.action_manager.get_term("joint_pos").velocity_commands
