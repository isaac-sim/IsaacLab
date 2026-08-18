# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for goal-conditioned cable routing."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg


def route_complete(env, command_name: str) -> torch.Tensor:
    """Terminate successful environments using the current cached route geometry."""
    command = env.command_manager.get_term(command_name)
    command.ensure_route_state_current(update_reward_delta=True)
    return command.succeeded


def robot_or_action_invalid(
    env,
    robot_cfgs: list[SceneEntityCfg],
    velocity_limit_factor: float = 2.0,
    joint_position_margin: float = 0.05,
) -> torch.Tensor:
    """Terminate non-finite actions or robot states that exceed authored joint limits."""
    if not math.isfinite(velocity_limit_factor) or not math.isfinite(joint_position_margin):
        raise ValueError("Robot-state termination thresholds must be finite.")
    if velocity_limit_factor <= 0.0 or joint_position_margin < 0.0:
        raise ValueError("velocity_limit_factor must be positive and joint_position_margin non-negative.")
    invalid = ~torch.isfinite(env.action_manager.action).all(dim=1)
    for asset_cfg in robot_cfgs:
        robot = env.scene[asset_cfg.name]
        position = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
        velocity = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
        position_limits = robot.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
        velocity_limits = robot.data.joint_vel_limits.torch[:, asset_cfg.joint_ids]
        body_pose = robot.data.body_link_pose_w.torch
        body_velocity = robot.data.body_link_vel_w.torch

        finite = (
            torch.isfinite(position).all(dim=1)
            & torch.isfinite(velocity).all(dim=1)
            & torch.isfinite(body_pose).all(dim=(1, 2))
            & torch.isfinite(body_velocity).all(dim=(1, 2))
        )
        outside_position = (
            (position < position_limits[..., 0] - joint_position_margin)
            | (position > position_limits[..., 1] + joint_position_margin)
        ).any(dim=1)
        outside_velocity = (velocity.abs() > velocity_limit_factor * velocity_limits).any(dim=1)
        invalid |= ~finite | outside_position | outside_velocity
    return invalid


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
