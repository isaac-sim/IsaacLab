# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for UR10 particle pushing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..ur10_particle_push_env import UR10ParticlePushEnv
    from ..ur10_particle_push_env_cfg import UR10ParticlePushEnvCfg


def build_bin_goal_mask(
    cfg: UR10ParticlePushEnvCfg,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Rasterize the calibrated bin region into the overhead heightmap frame."""
    height_cells, width_cells = cfg.heightmap_shape
    x_lo, x_hi = cfg.heightmap_x_bounds
    y_lo, y_hi = cfg.heightmap_y_bounds
    x = torch.linspace(
        x_lo + 0.5 * (x_hi - x_lo) / width_cells,
        x_hi - 0.5 * (x_hi - x_lo) / width_cells,
        width_cells,
        device=device,
    )
    y = torch.linspace(
        y_lo + 0.5 * (y_hi - y_lo) / height_cells,
        y_hi - 0.5 * (y_hi - y_lo) / height_cells,
        height_cells,
        device=device,
    )
    inside_x = (x >= cfg.bin_inner_x_bounds[0]) & (x <= cfg.bin_inner_x_bounds[1])
    inside_y = (y >= cfg.bin_inner_y_bounds[0]) & (y <= cfg.bin_inner_y_bounds[1])
    return (inside_y[:, None] & inside_x[None, :]).float()


def policy_observation(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return robot and task observations for the actor."""
    return env.policy_observation()


def heightmap_observation(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return current/delayed granular surfaces and the calibrated bin mask."""
    return env.heightmap_observation()


def critic_observation(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return exact particle summaries available only to the asymmetric critic."""
    return env.critic_observation()
