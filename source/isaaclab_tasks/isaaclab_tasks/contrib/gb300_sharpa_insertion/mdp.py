# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured connector, cable and spring-clip state for observations and insertion rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
)

from . import socket_geometry
from .workcell import head_goal_for_socket

if TYPE_CHECKING:
    from .env import GB300SharpaInsertionEnv


def tcp_pose(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return the canonical palm world pose [m, XYZW], shape [N, 7]."""
    return env.scene["robot"].data.body_pose_w.torch[:, env.tcp_body_id]


def head_pose(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return physical connector world pose [m, XYZW], not its rod attachment proxy."""
    if env.cfg.hydroelastic_contacts:
        return env.scene["connector"].data.body_link_pose_w.torch[:, env.head_body_id]
    return env.scene["cable"].data.segment_pose_w.torch[:, 0]


def head_state(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return the loose connector pose in the robot base frame [m, XYZW]."""
    root = env.scene["robot"].data.root_pose_w.torch
    head = head_pose(env)
    position, rotation = subtract_frame_transforms(root[:, :3], root[:, 3:], head[:, :3], head[:, 3:])
    return torch.cat((position, rotation), dim=-1)


def goal_state(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return the target seating pose in the robot base frame [m, XYZW]."""
    root = env.scene["robot"].data.root_pose_w.torch
    goal = root.new_tensor((*head_goal_for_socket(env.cfg.target_socket_position_e), 2**-0.5, 0.0, 0.0, 2**-0.5))
    position = goal[:3].expand(env.num_envs, -1) + env.scene.env_origins
    rotation = goal[3:].expand(env.num_envs, -1)
    position, rotation = subtract_frame_transforms(root[:, :3], root[:, 3:], position, rotation)
    return torch.cat((position, rotation), dim=-1)


def latch_state(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return clip deflection [rad] and hinge angular velocity [rad/s], shape [N, 2].

    Negative angles compress the clip toward the plug. The measured two-body
    state has the same semantics for native VBD and articulated MJWarp clips.
    """
    head = head_pose(env)
    pose, angular_velocity = _latch_body_state(env)
    relative = quat_unique(quat_mul(quat_conjugate(head[:, 3:]), pose[:, 3:]))
    angle = 2.0 * torch.atan2(-relative[:, 0], relative[:, 3])
    head_angular_velocity = (
        env.scene["connector"].data.body_com_ang_vel_w.torch[:, env.head_body_id]
        if env.cfg.hydroelastic_contacts
        else env.scene["cable"].data.segment_velocity_w.torch[:, 0, 3:]
    )
    axis = head.new_tensor((-1.0, 0.0, 0.0)).expand(env.num_envs, -1)
    rate = ((angular_velocity - head_angular_velocity) * quat_apply(head[:, 3:], axis)).sum(-1)
    return torch.stack((angle, rate), dim=-1)


def latch_capture_margins(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return signed axial, lateral and vertical latch capture margins [m]."""
    return socket_geometry.latch_engagement_margins(_latch_pose_in_socket(env))


def latch_captured(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Detect keeper capture, allowing 50 µm loaded-contact compliance."""
    margins = latch_capture_margins(env)
    return (margins[:, 0] > -0.00005) & (margins[:, 1:] > 0.0).all(-1)


def latch_metrics(env: GB300SharpaInsertionEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return capture state and capture/release/exit margins from one measured latch pose.

    Returns:
        Captured flags, shape [N], capture margins [m], shape [N, 3], release
        margins [m], shape [N], and keeper-exit margins [m], shape [N].
    """
    margins, release, exit_margin = socket_geometry.latch_geometry_margins(_latch_pose_in_socket(env))
    captured = (margins[:, 0] > -0.00005) & (margins[:, 1:] > 0.0).all(-1)
    return captured, margins, release, exit_margin


def head_seated(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return whether the physical head is within 3 [mm] and 0.08 [rad] of its seated pose."""
    head = head_state(env)
    target = goal_state(env)
    position_error = torch.linalg.vector_norm(head[:, :3] - target[:, :3], dim=-1)
    angle_error = quat_error_magnitude(head[:, 3:], target[:, 3:])
    return (position_error < 0.003) & (angle_error < 0.08)


def insertion_success(
    env: GB300SharpaInsertionEnv, require_release: bool = True, release_clearance: float = 0.19
) -> torch.Tensor:
    """Detect a seated, captured connector and physical hand withdrawal.

    Args:
        env: Environment containing the physical connector and socket.
        require_release: Require the palm to withdraw beyond the finger reach envelope.
        release_clearance: Minimum palm-to-connector separation [m] after release.

    Returns:
        Boolean insertion success for each environment. No scripted finger pose is required.
    """
    seated = head_seated(env) & latch_captured(env)
    if not require_release:
        return seated
    distance = torch.linalg.vector_norm(tcp_pose(env)[:, :3] - head_pose(env)[:, :3], dim=-1)
    return seated & (distance > release_clearance)


def reaching(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Reward palm proximity to the physical connector, with zero for invalid state."""
    distance = torch.linalg.vector_norm(tcp_pose(env)[:, :3] - head_pose(env)[:, :3], dim=-1)
    return torch.nan_to_num(1.0 - torch.tanh(distance / 0.1), nan=0.0)


def head_goal_position(env: GB300SharpaInsertionEnv, std: float = 0.05) -> torch.Tensor:
    """Reward connector-to-seat distance using a bounded tanh kernel with scale [m]."""
    distance = torch.linalg.vector_norm(head_state(env)[:, :3] - goal_state(env)[:, :3], dim=-1)
    return torch.nan_to_num(1.0 - torch.tanh(distance / std), nan=0.0)


def head_goal_orientation(env: GB300SharpaInsertionEnv, std: float = 0.25) -> torch.Tensor:
    """Reward measured connector alignment using a bounded angular kernel with scale [rad]."""
    angle = quat_error_magnitude(head_state(env)[:, 3:], goal_state(env)[:, 3:])
    return torch.nan_to_num(1.0 - torch.tanh(angle / std), nan=0.0)


def cable_invalid(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Detect non-finite state or a cable falling below 0.05 [m] above its ground."""
    poses = env.scene["cable"].data.segment_pose_w.torch
    heights = poses[..., 2] - env.scene.env_origins[:, None, 2]
    return (
        ~torch.isfinite(poses).all(dim=(1, 2))
        | (heights < 0.05).any(dim=1)
        | ~torch.isfinite(head_pose(env)).all(dim=1)
        | ~torch.isfinite(latch_state(env)).all(dim=1)
    )


def _latch_body_state(env: GB300SharpaInsertionEnv) -> tuple[torch.Tensor, torch.Tensor]:
    if env.cfg.hydroelastic_contacts:
        connector = env.scene["connector"].data
        return connector.body_link_pose_w.torch[:, env.latch_body_id], connector.body_com_ang_vel_w.torch[
            :, env.latch_body_id
        ]
    latch = env.scene["latch"].data
    return latch.body_link_pose_w.torch[:, 0], latch.body_com_ang_vel_w.torch[:, 0]


def _latch_pose_in_socket(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    latch, _ = _latch_body_state(env)
    socket_position = latch.new_tensor(env.cfg.target_socket_position_e) + env.scene.env_origins
    return torch.cat((latch[:, :3] - socket_position, latch[:, 3:]), dim=-1)
