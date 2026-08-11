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
    active = torch.nn.functional.one_hot(command.active_step, num_classes=command.cfg.max_route_steps).float()
    route_length = command.valid_steps.sum(dim=1, keepdim=True).float() / command.cfg.max_route_steps
    return torch.cat(
        [
            active,
            command.completed_steps.float(),
            command.directed_progress,
            route_length,
            command.succeeded[:, None].float(),
        ],
        dim=-1,
    )


def active_goal_geometry(
    env,
    command_name: str,
    cable_cfg: SceneEntityCfg,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ground the active route token in gripper and cable geometry [m]."""
    command = env.command_manager.get_term(command_name)
    target = command.active_peg_positions_w
    cable_points = env.scene[cable_cfg.name].data.segment_pose_w.torch[..., :3]
    left_robot = env.scene[left_ee_cfg.name]
    right_robot = env.scene[right_ee_cfg.name]
    left_ee = yam_contact_frame_position_w(left_robot, left_ee_cfg.body_ids[0])
    right_ee = yam_contact_frame_position_w(right_robot, right_ee_cfg.body_ids[0])

    endpoint_error = cable_points[:, (0, -1)] - target[:, None, :]
    cable_distance = torch.linalg.vector_norm(cable_points - target[:, None, :], dim=-1).amin(dim=1, keepdim=True)
    left_cable_distance = torch.linalg.vector_norm(cable_points - left_ee[:, None, :], dim=-1).amin(dim=1, keepdim=True)
    right_cable_distance = torch.linalg.vector_norm(cable_points - right_ee[:, None, :], dim=-1).amin(
        dim=1, keepdim=True
    )
    return torch.cat(
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
    return torch.cat([positions_b, velocities, tangent], dim=-1).flatten(start_dim=1)
