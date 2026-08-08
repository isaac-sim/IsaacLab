# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms and particle outcome metrics for UR10 particle pushing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..ur10_particle_push_env import UR10ParticlePushEnv
    from ..ur10_particle_push_env_cfg import UR10ParticlePushEnvCfg


def update_success_streak(
    previous_streak: torch.Tensor,
    bin_fraction: torch.Tensor,
    spill_fraction: torch.Tensor,
    *,
    success_fraction: float | torch.Tensor,
    max_spill_fraction: float,
    dwell_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a persistent delivery predicate that rejects transient bin occupancy."""
    qualifies = (bin_fraction >= success_fraction) & (spill_fraction <= max_spill_fraction)
    next_streak = torch.where(qualifies, previous_streak + 1, torch.zeros_like(previous_streak))
    return next_streak, next_streak >= dwell_steps


def compute_particle_bin_mask(
    particle_position_e: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
) -> torch.Tensor:
    """Return particles delivered below the bin rim."""
    x, y, z = particle_position_e.unbind(dim=-1)
    return (
        (x >= cfg.bin_inner_x_bounds[0])
        & (x <= cfg.bin_inner_x_bounds[1])
        & (y >= cfg.bin_inner_y_bounds[0])
        & (y <= cfg.bin_inner_y_bounds[1])
        & (z >= cfg.bin_inner_z_bounds[0])
        & (z <= cfg.bin_inner_z_bounds[1])
    )


def compute_particle_metrics(
    particle_position_e: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute terminal delivery and irreversible-spill fractions from particle positions."""
    x, y, z = particle_position_e.unbind(dim=-1)
    delivered_below_rim = compute_particle_bin_mask(particle_position_e, cfg)
    bin_fraction = delivered_below_rim.float().mean(dim=1)

    in_physical_bin = (
        (x >= cfg.bin_inner_x_bounds[0])
        & (x <= cfg.bin_inner_x_bounds[1])
        & (y >= cfg.bin_inner_y_bounds[0])
        & (y <= cfg.bin_inner_y_bounds[1])
        & (z >= cfg.bin_physical_z_bounds[0])
        & (z <= cfg.bin_physical_z_bounds[1])
    )
    below_table = (z < -0.03) & ~in_physical_bin
    outside_lateral_workspace = (y.abs() > 0.455) | (x < -0.2461) | (x > cfg.bin_inner_x_bounds[1] + 0.05)
    spill_fraction = (below_table | outside_lateral_workspace).float().mean(dim=1)
    return bin_fraction, spill_fraction


def invalid_state(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Terminate numerical failures or extreme rigid/particle state."""
    env.update_task_state()
    return env.invalid_state


def escaped_workspace(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Terminate when too much media leaves the bounded sparse-grid workspace."""
    env.update_task_state()
    return env.escaped_workspace


def excessive_spill(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Terminate an irrecoverable spill."""
    env.update_task_state()
    return env.excessive_spill


def success(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Terminate after sustained delivery at the active target fraction."""
    env.update_task_state()
    return env.success_this_step


def time_out(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Truncate at the deadline only when no terminal success or failure occurred."""
    env.update_task_state()
    terminated = env.success_this_step | env.invalid_state | env.escaped_workspace | env.excessive_spill
    return (env.episode_length_buf >= env.max_episode_length) & ~terminated
