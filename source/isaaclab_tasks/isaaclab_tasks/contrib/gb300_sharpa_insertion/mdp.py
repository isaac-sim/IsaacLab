# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured connector, cable and spring-clip state for the physical example."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_unique,
)

from . import socket_geometry

if TYPE_CHECKING:
    from .env import GB300SharpaInsertionEnv


def tcp_pose(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return the canonical palm world pose [m, XYZW], shape [N, 7]."""
    return env.scene["robot"].data.body_pose_w.torch[:, env.tcp_body_id]


def head_pose(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return physical connector world pose [m, XYZW], not its rod attachment proxy."""
    return env.scene["connector"].data.body_link_pose_w.torch[:, env.head_body_id]


def cable_state(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return measured segment poses [m, XYZW], flattened to shape [N, S * 7]."""
    return env.scene["cable"].data.segment_pose_w.torch.flatten(start_dim=1)


def latch_state(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    """Return clip deflection [rad] and hinge angular velocity [rad/s], shape [N, 2].

    Negative angles compress the clip toward the plug. The relative head/clip
    pose and angular velocity measure the articulated MJWarp spring.
    """
    head = head_pose(env)
    pose, angular_velocity = _latch_body_state(env)
    relative = quat_unique(quat_mul(quat_conjugate(head[:, 3:]), pose[:, 3:]))
    angle = 2.0 * torch.atan2(-relative[:, 0], relative[:, 3])
    head_angular_velocity = env.scene["connector"].data.body_com_ang_vel_w.torch[:, env.head_body_id]
    axis = head.new_tensor((-1.0, 0.0, 0.0)).expand(env.num_envs, -1)
    rate = ((angular_velocity - head_angular_velocity) * quat_apply(head[:, 3:], axis)).sum(-1)
    return torch.stack((angle, rate), dim=-1)


def latch_metrics(env: GB300SharpaInsertionEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return capture state and capture/release/exit margins from one measured latch pose.

    Returns:
        Captured flags, shape [N], capture margins [m], shape [N, 3], release
        margins [m], shape [N], and keeper-exit margins [m], shape [N].
    """
    margins, release, exit_margin = socket_geometry.latch_geometry_margins(_latch_pose_in_socket(env))
    captured = (margins[:, 0] > -0.00005) & (margins[:, 1:] > 0.0).all(-1)
    return captured, margins, release, exit_margin


def _latch_body_state(env: GB300SharpaInsertionEnv) -> tuple[torch.Tensor, torch.Tensor]:
    connector = env.scene["connector"].data
    return connector.body_link_pose_w.torch[:, env.latch_body_id], connector.body_com_ang_vel_w.torch[
        :, env.latch_body_id
    ]


def _latch_pose_in_socket(env: GB300SharpaInsertionEnv) -> torch.Tensor:
    latch, _ = _latch_body_state(env)
    socket_position = latch.new_tensor(env.cfg.target_socket_position_e) + env.scene.env_origins
    return torch.cat((latch[:, :3] - socket_position, latch[:, 3:]), dim=-1)
