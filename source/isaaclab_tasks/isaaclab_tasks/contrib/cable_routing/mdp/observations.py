# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for goal-conditioned cable routing."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from ..yam_frames import yam_contact_frame_position_w


def route_task_state(env, command_name: str) -> torch.Tensor:
    """Return active-step, completion, and directed-winding state."""
    command = env.command_manager.get_term(command_name)
    command.ensure_route_state_current()
    active = torch.nn.functional.one_hot(command.active_step, num_classes=command.cfg.max_route_steps).float()
    route_length = command.valid_steps.sum(dim=1, keepdim=True).float() / command.cfg.max_route_steps
    state = torch.cat(
        [
            active,
            command.completed_steps.float(),
            command.directed_progress,
            route_length,
            command.succeeded[:, None].float(),
        ],
        dim=-1,
    )
    return torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


def active_goal_geometry(
    env,
    command_name: str,
    cable_cfg: SceneEntityCfg,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ground the active route token in gripper and cable geometry [m]."""
    command = env.command_manager.get_term(command_name)
    command.ensure_route_state_current()
    target = torch.nan_to_num(command.active_peg_positions_w, nan=0.0, posinf=0.0, neginf=0.0)
    cable_points = torch.nan_to_num(
        env.scene[cable_cfg.name].data.segment_pose_w.torch[..., :3], nan=0.0, posinf=0.0, neginf=0.0
    )
    left_robot = env.scene[left_ee_cfg.name]
    right_robot = env.scene[right_ee_cfg.name]
    left_ee = torch.nan_to_num(
        yam_contact_frame_position_w(left_robot, left_ee_cfg.body_ids[0]), nan=0.0, posinf=0.0, neginf=0.0
    )
    right_ee = torch.nan_to_num(
        yam_contact_frame_position_w(right_robot, right_ee_cfg.body_ids[0]), nan=0.0, posinf=0.0, neginf=0.0
    )

    endpoint_error = cable_points[:, (0, -1)] - target[:, None, :]
    cable_distance = torch.linalg.vector_norm(cable_points - target[:, None, :], dim=-1).amin(dim=1, keepdim=True)
    left_cable_distance = torch.linalg.vector_norm(cable_points - left_ee[:, None, :], dim=-1).amin(dim=1, keepdim=True)
    right_cable_distance = torch.linalg.vector_norm(cable_points - right_ee[:, None, :], dim=-1).amin(
        dim=1, keepdim=True
    )
    geometry = torch.cat(
        [
            target - left_ee,
            target - right_ee,
            endpoint_error.flatten(start_dim=1),
            cable_distance,
            left_cable_distance,
            right_cable_distance,
        ],
        dim=-1,
    )
    return torch.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0)


def finite_last_action(env, action_name: str | None = None) -> torch.Tensor:
    """Return the bounded action that the task's action terms can apply."""
    action = env.action_manager.action if action_name is None else env.action_manager.get_term(action_name).raw_actions
    return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def finite_joint_pos_rel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return limit-bounded relative joint positions with invalid values replaced by defaults."""
    asset = env.scene[asset_cfg.name]
    position = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    position = torch.where(torch.isfinite(position), position, default)
    position = torch.maximum(torch.minimum(position, limits[..., 1]), limits[..., 0])
    return position - default


def finite_joint_vel_rel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return bounded relative joint velocities with invalid values replaced by defaults."""
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_vel.torch[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_vel_limits.torch[:, asset_cfg.joint_ids]
    velocity = torch.where(torch.isfinite(velocity), velocity, default)
    velocity = torch.maximum(torch.minimum(velocity, 2.0 * limits), -2.0 * limits)
    return velocity - default


def sampled_cable_state_b(env, asset_cfg: SceneEntityCfg, num_samples: int = 32) -> torch.Tensor:
    """Return sampled cable position, linear velocity, and tangent in each environment frame."""
    cable = env.scene[asset_cfg.name]
    positions_w = cable.data.segment_pose_w.torch[..., :3]
    linear_velocity_w = cable.data.segment_velocity_w.torch[..., :3]
    sample_ids = (
        torch.linspace(
            0,
            positions_w.shape[1] - 1,
            num_samples,
            device=positions_w.device,
        )
        .round()
        .long()
    )
    positions_b = positions_w[:, sample_ids] - env.scene.env_origins[:, None, :]
    velocities = linear_velocity_w[:, sample_ids]

    # Compute only the selected forward differences. The final sample retains
    # the previous segment's tangent, matching the full-chain implementation.
    tangent_start_ids = sample_ids.clamp(max=positions_w.shape[1] - 2)
    tangent_end_ids = tangent_start_ids + 1
    tangent = torch.nn.functional.normalize(
        positions_w[:, tangent_end_ids] - positions_w[:, tangent_start_ids], dim=-1, eps=1.0e-8
    )
    state = torch.cat([positions_b, velocities, tangent], dim=-1).flatten(start_dim=1)
    return torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
