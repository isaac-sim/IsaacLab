# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure state predicates shared by the Franka Pour environment and its MDP terms."""

from __future__ import annotations

import torch


def state_finite(
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    tcp_body_q: torch.Tensor,
    cup_body_q: torch.Tensor,
    cup_lin_vel: torch.Tensor,
    cup_ang_vel: torch.Tensor,
    particle_pos: torch.Tensor,
) -> torch.Tensor:
    """Return a per-environment finite-state mask using unsanitized simulation tensors."""
    robot_ok = torch.isfinite(robot_joint_pos).all(dim=-1) & torch.isfinite(robot_joint_vel).all(dim=-1)
    tcp_ok = torch.isfinite(tcp_body_q).all(dim=-1)
    cup_ok = (
        torch.isfinite(cup_body_q).all(dim=-1)
        & torch.isfinite(cup_lin_vel).all(dim=-1)
        & torch.isfinite(cup_ang_vel).all(dim=-1)
    )
    media_ok = torch.isfinite(particle_pos).all(dim=(1, 2))
    return robot_ok & tcp_ok & cup_ok & media_ok


def rigid_state_in_bounds(
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    joint_pos_limits: torch.Tensor,
    tcp_body_q: torch.Tensor,
    cup_body_q: torch.Tensor,
    cup_lin_vel: torch.Tensor,
    cup_ang_vel: torch.Tensor,
    env_origins: torch.Tensor,
    lower_bound: tuple[float, float, float] | torch.Tensor,
    upper_bound: tuple[float, float, float] | torch.Tensor,
    joint_position_margin: float,
    max_joint_velocity: float,
    max_cup_linear_velocity: float,
    max_cup_angular_velocity: float,
) -> torch.Tensor:
    """Return whether every rigid state that feeds actor observations is physically bounded."""
    lower = torch.as_tensor(lower_bound, device=robot_joint_pos.device, dtype=robot_joint_pos.dtype)
    upper = torch.as_tensor(upper_bound, device=robot_joint_pos.device, dtype=robot_joint_pos.dtype)
    joint_lower = joint_pos_limits[..., 0] - float(joint_position_margin)
    joint_upper = joint_pos_limits[..., 1] + float(joint_position_margin)
    joint_position_ok = ((robot_joint_pos >= joint_lower) & (robot_joint_pos <= joint_upper)).all(dim=-1)
    joint_velocity_ok = (torch.abs(robot_joint_vel) <= float(max_joint_velocity)).all(dim=-1)

    tcp_position = tcp_body_q[:, :3] - env_origins
    cup_position = cup_body_q[:, :3] - env_origins
    tcp_position_ok = ((tcp_position >= lower) & (tcp_position <= upper)).all(dim=-1)
    cup_position_ok = ((cup_position >= lower) & (cup_position <= upper)).all(dim=-1)
    tcp_quat_norm = torch.linalg.vector_norm(tcp_body_q[:, 3:7], dim=-1)
    cup_quat_norm = torch.linalg.vector_norm(cup_body_q[:, 3:7], dim=-1)
    pose_ok = (
        torch.isfinite(tcp_body_q).all(dim=-1)
        & torch.isfinite(cup_body_q).all(dim=-1)
        & (torch.abs(tcp_quat_norm - 1.0) <= 0.1)
        & (torch.abs(cup_quat_norm - 1.0) <= 0.1)
    )

    cup_linear_velocity_ok = torch.linalg.vector_norm(cup_lin_vel, dim=-1) <= float(max_cup_linear_velocity)
    cup_angular_velocity_ok = torch.linalg.vector_norm(cup_ang_vel, dim=-1) <= float(max_cup_angular_velocity)
    return (
        joint_position_ok
        & joint_velocity_ok
        & tcp_position_ok
        & cup_position_ok
        & pose_ok
        & cup_linear_velocity_ok
        & cup_angular_velocity_ok
    )


def particles_in_workspace(
    particle_pos_e: torch.Tensor,
    lower_bound: tuple[float, float, float] | torch.Tensor,
    upper_bound: tuple[float, float, float] | torch.Tensor,
) -> torch.Tensor:
    """Return whether every particle lies inside its environment-local workspace."""
    lower = torch.as_tensor(lower_bound, device=particle_pos_e.device, dtype=particle_pos_e.dtype)
    upper = torch.as_tensor(upper_bound, device=particle_pos_e.device, dtype=particle_pos_e.dtype)
    return ((particle_pos_e >= lower) & (particle_pos_e <= upper)).all(dim=(1, 2))


def spilled_particle_mask(
    particle_pos_e: torch.Tensor,
    in_source: torch.Tensor,
    in_target: torch.Tensor,
    max_height: float,
) -> torch.Tensor:
    """Classify particles outside both cups that have reached the table [m]."""
    outside_cups = ~in_source & ~in_target
    return outside_cups & (particle_pos_e[..., 2] <= float(max_height))


def delivered_particle_mask(in_source: torch.Tensor, in_target: torch.Tensor) -> torch.Tensor:
    """Classify particles inside the receiver only after they have left the source cup."""
    return in_target & ~in_source
