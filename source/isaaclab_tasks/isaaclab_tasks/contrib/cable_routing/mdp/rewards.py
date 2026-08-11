# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for goal-conditioned cable routing."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from ..yam_frames import yam_contact_frame_position_w
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
    return env.command_manager.get_term(command_name).route_progress_delta


def route_success(env, command_name: str) -> torch.Tensor:
    """Return one for environments that have completed their sampled route."""
    return env.command_manager.get_term(command_name).succeeded.float()


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
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward both grippers for approaching the cable."""
    points = env.scene[cable_cfg.name].data.segment_pose_w.torch[..., :3]
    left = yam_contact_frame_position_w(env.scene[left_ee_cfg.name], left_ee_cfg.body_ids[0])
    right = yam_contact_frame_position_w(env.scene[right_ee_cfg.name], right_ee_cfg.body_ids[0])
    left_distance = torch.linalg.vector_norm(points - left[:, None, :], dim=-1).amin(dim=1)
    right_distance = torch.linalg.vector_norm(points - right[:, None, :], dim=-1).amin(dim=1)
    reward = 1.0 - 0.5 * (torch.tanh(left_distance / std) + torch.tanh(right_distance / std))
    return _zero_invalid_reward_rows(reward, _finite_rows(points, left, right))


def cable_stretch(env, cable_cfg: SceneEntityCfg, rest_length: float) -> torch.Tensor:
    """Return mean squared relative cable-joint endpoint gap."""
    poses = env.scene[cable_cfg.name].data.segment_pose_w.torch
    reward = torch.square(cable_relative_joint_gap(poses, rest_length)).mean(dim=1)
    return _zero_invalid_reward_rows(reward, _finite_rows(poses))
