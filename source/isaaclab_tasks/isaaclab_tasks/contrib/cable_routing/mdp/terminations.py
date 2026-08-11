# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for goal-conditioned cable routing."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def route_complete(env, command_name: str) -> torch.Tensor:
    """Refresh route geometry and terminate successful environments."""
    command = env.command_manager.get_term(command_name)
    command.refresh_route_state(update_reward_delta=True)
    return command.succeeded


def cable_invalid_or_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg,
    xy_limit: tuple[float, float] = (0.55, 0.45),
    z_range: tuple[float, float] = (0.0, 1.4),
) -> torch.Tensor:
    """Terminate non-finite cables or cables that leave the tabletop workspace."""
    cable = env.scene[asset_cfg.name]
    poses = cable.data.segment_pose_w.torch
    velocities = cable.data.segment_velocity_w.torch
    points = poses[..., :3]
    points_b = points - env.scene.env_origins[:, None, :]
    finite = torch.isfinite(poses).all(dim=(1, 2)) & torch.isfinite(velocities).all(dim=(1, 2))
    within_xy = (points_b[..., 0].abs() <= xy_limit[0]) & (points_b[..., 1].abs() <= xy_limit[1])
    within_z = (points_b[..., 2] >= z_range[0]) & (points_b[..., 2] <= z_range[1])
    return ~finite | ~within_xy.all(dim=1) | ~within_z.all(dim=1)
