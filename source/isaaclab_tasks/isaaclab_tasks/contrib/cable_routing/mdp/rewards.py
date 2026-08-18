# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for goal-conditioned cable routing."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg

from ..frames import contact_frame_position_w
from .actions import canonical_task_actions
from .cable_geometry import cable_relative_joint_gap


def _finite_rows(*values: torch.Tensor) -> torch.Tensor:
    """Return rows whose complete reward geometry is finite."""
    if not values:
        raise ValueError("At least one tensor is required to evaluate finite rows.")
    finite = torch.ones(values[0].shape[0], dtype=torch.bool, device=values[0].device)
    for value in values:
        if value.shape[0] != finite.shape[0]:
            raise ValueError("All reward tensors must contain the same number of environments.")
        finite &= torch.isfinite(value).flatten(start_dim=1).all(dim=1)
    return finite


def _zero_invalid_reward_rows(reward: torch.Tensor, finite_geometry: torch.Tensor) -> torch.Tensor:
    """Zero terminal numerical-failure rows while preserving finite reward rows."""
    valid = finite_geometry & torch.isfinite(reward)
    return torch.where(valid, reward, torch.zeros_like(reward))


def route_progress(env, command_name: str) -> torch.Tensor:
    """Reward the per-policy-step increase in ordered route progress."""
    command = env.command_manager.get_term(command_name)
    command.ensure_route_state_current(update_reward_delta=True)
    return command.route_progress_delta


def route_success(
    env,
    command_name: str,
    failure_termination_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Return a unit-integral pulse for valid terminal route success."""
    command = env.command_manager.get_term(command_name)
    command.ensure_route_state_current(update_reward_delta=True)
    succeeded = command.succeeded.clone()
    for term_name in failure_termination_names:
        succeeded &= ~env.termination_manager.get_term(term_name)
    return succeeded.float() / max(float(env.step_dt), 1.0e-8)


def route_failure(env, termination_names: tuple[str, ...]) -> torch.Tensor:
    """Return one unit-integral failure pulse for invalid terminal transitions."""
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in termination_names:
        failed |= env.termination_manager.get_term(term_name)
    return failed.float() / max(float(env.step_dt), 1.0e-8)


def finite_action_rate_l2(env, binary_action_names: tuple[str, ...] = ()) -> torch.Tensor:
    """Penalize changes in finite arm actions and binary gripper command states."""
    action = canonical_task_actions(env, env.action_manager.action, binary_action_names)
    previous = canonical_task_actions(env, env.action_manager.prev_action, binary_action_names)
    return torch.square(action - previous).sum(dim=1)


def finite_joint_vel_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocity without allowing failed rows to contaminate rewards."""
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_vel_limits.torch[:, asset_cfg.joint_ids]
    finite = torch.isfinite(velocity).all(dim=1)
    velocity = torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    velocity = torch.maximum(torch.minimum(velocity, 2.0 * limits), -2.0 * limits)
    reward = torch.square(velocity).sum(dim=1)
    return torch.where(finite & torch.isfinite(reward), reward, torch.zeros_like(reward))


def cable_near_active_peg(env, command_name: str, cable_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward any cable segment approaching the active peg."""
    command = env.command_manager.get_term(command_name)
    points = env.scene[cable_cfg.name].data.segment_pose_w.torch[..., :3]
    target = command.active_peg_positions_w
    distance = torch.linalg.vector_norm(points - target[:, None, :], dim=-1).amin(dim=1)
    reward = 1.0 - torch.tanh(distance / std)
    return _zero_invalid_reward_rows(reward, _finite_rows(points, target))


def grippers_near_cable(
    env,
    cable_cfg: SceneEntityCfg,
    end_effector_cfgs: Sequence[SceneEntityCfg],
    contact_frame_offset_positions: Sequence[tuple[float, float, float]],
    std: float,
) -> torch.Tensor:
    """Reward both configured contact frames for approaching the cable."""
    points = env.scene[cable_cfg.name].data.segment_pose_w.torch[..., :3]
    contact_positions = [
        contact_frame_position_w(env.scene[ee_cfg.name], ee_cfg.body_ids[0], offset_position)
        for ee_cfg, offset_position in zip(end_effector_cfgs, contact_frame_offset_positions, strict=True)
    ]
    left, right = contact_positions
    left_distance = torch.linalg.vector_norm(points - left[:, None, :], dim=-1).amin(dim=1)
    right_distance = torch.linalg.vector_norm(points - right[:, None, :], dim=-1).amin(dim=1)
    reward = 1.0 - 0.5 * (torch.tanh(left_distance / std) + torch.tanh(right_distance / std))
    return _zero_invalid_reward_rows(reward, _finite_rows(points, left, right))


def cable_stretch(env, cable_cfg: SceneEntityCfg, rest_length: float) -> torch.Tensor:
    """Return mean squared relative cable-joint endpoint gap."""
    poses = env.scene[cable_cfg.name].data.segment_pose_w.torch
    reward = torch.square(cable_relative_joint_gap(poses, rest_length)).mean(dim=1)
    return _zero_invalid_reward_rows(reward, _finite_rows(poses))
