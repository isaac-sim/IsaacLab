# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .utils import tail_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def tcp_position_w(
    robot: Articulation,
    body_ids: Sequence[int] | torch.Tensor | slice,
    body_offsets: tuple[tuple[float, float, float], ...] | torch.Tensor,
) -> torch.Tensor:
    """Return the mean of body-local surface points transformed into world coordinates [m], shape [N, 3]."""
    position = robot.data.body_pos_w.torch[:, body_ids]
    quaternion = robot.data.body_quat_w.torch[:, body_ids]
    offsets = torch.as_tensor(body_offsets, device=position.device, dtype=position.dtype).expand_as(position)
    return (position + math_utils.quat_apply(quaternion, offsets)).mean(dim=1)


def tcp_velocity_w(
    robot: Articulation,
    body_ids: Sequence[int] | torch.Tensor | slice,
    body_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """Return the mean surface-point velocity, including each finger's angular motion [m/s], shape [N, 3]."""
    velocity = robot.data.body_link_vel_w.torch[:, body_ids]
    quaternion = robot.data.body_quat_w.torch[:, body_ids]
    offsets = velocity.new_tensor(body_offsets).expand_as(velocity[..., :3])
    lever = math_utils.quat_apply(quaternion, offsets)
    return (velocity[..., :3] + torch.linalg.cross(velocity[..., 3:], lever, dim=-1)).mean(dim=1)


def finger_tail_signed_distance(env: ManagerBasedEnv) -> torch.Tensor:
    """Return signed surface separation for both fingers of each gripper [m].

    The columns are left-gripper left/right finger followed by right-gripper left/right finger.
    A positive capped value denotes no collision candidate, zero denotes touching, and negative values
    denote contact-solver penetration.
    """
    signed_distance = env.scene["finger_tail_contacts"].data
    if signed_distance is None:
        raise RuntimeError("Shoelace finger-tail contact observation is not initialized")
    return signed_distance


def finger_point_state_w(
    env: ManagerBasedEnv, finger_cfgs: tuple[SceneEntityCfg, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four surface reference points and their linear velocities [m, m/s], shape [N, 2, 2, 3]."""
    positions, velocities = [], []
    for cfg, offset in zip(finger_cfgs, env.cfg.finger_body_offsets, strict=True):
        robot = env.scene[cfg.name]
        position = robot.data.body_pos_w.torch[:, cfg.body_ids].squeeze(1)
        quaternion = robot.data.body_quat_w.torch[:, cfg.body_ids].squeeze(1)
        velocity = robot.data.body_link_vel_w.torch[:, cfg.body_ids].squeeze(1)
        lever = math_utils.quat_apply(quaternion, position.new_tensor(offset).expand_as(position))
        positions.append(position + lever)
        velocities.append(velocity[:, :3] + torch.linalg.cross(velocity[:, 3:], lever, dim=-1))
    shape = (env.num_envs, 2, 2, 3)
    return torch.stack(positions, dim=1).reshape(shape), torch.stack(velocities, dim=1).reshape(shape)


def pad_to_tail(
    env: ManagerBasedEnv,
    finger_cfgs: tuple[SceneEntityCfg, ...],
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return each surface-point-to-tail vector in its robot root frame [m], shape [N, 12]."""
    points, _ = finger_point_state_w(env, finger_cfgs)
    tails, _ = tail_state(env, cable_cfgs)
    vectors = []
    for arm in range(2):
        quaternion = env.scene[finger_cfgs[2 * arm].name].data.root_quat_w.torch[:, None].expand(-1, 2, -1)
        vectors.append(math_utils.quat_apply_inverse(quaternion, tails[:, arm, None] - points[:, arm]))
    return torch.stack(vectors, dim=1).flatten(start_dim=1)


def finger_tail_relative_velocity(
    env: ManagerBasedEnv,
    finger_cfgs: tuple[SceneEntityCfg, ...],
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return tail-minus-surface-point velocities separately in robot root frames [m/s], shape [N, 12]."""
    _, velocities = finger_point_state_w(env, finger_cfgs)
    _, tail_velocities = tail_state(env, cable_cfgs)
    vectors = []
    for arm in range(2):
        quaternion = env.scene[finger_cfgs[2 * arm].name].data.root_quat_w.torch[:, None].expand(-1, 2, -1)
        vectors.append(math_utils.quat_apply_inverse(quaternion, tail_velocities[:, arm, None] - velocities[:, arm]))
    return torch.stack(vectors, dim=1).flatten(start_dim=1)


def gripper_closing_tracking_error(
    env: ManagerBasedEnv,
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    closing_direction: float = -1.0,
) -> torch.Tensor:
    """Return current joint-target tracking errors only in the closing direction [m or rad].

    Args:
        env: Task environment.
        robot_cfgs: Driven finger joints for each arm.
        closing_direction: Sign of joint motion when closing; -1 for Franka and +1 for Inspire.
    """
    errors = []
    for robot_cfg in robot_cfgs:
        robot = env.scene[robot_cfg.name]
        actual = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
        target = robot.data.joint_pos_target.torch[:, robot_cfg.joint_ids]
        errors.append((closing_direction * (target - actual)).clamp_min_(0.0))
    return torch.cat(errors, dim=-1)


def tails_to_tcp(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return tail-to-TCP vectors in the controlling robot root frames [m]."""
    tail_positions, _ = tail_state(env, cable_cfgs)
    vectors = []
    for arm, robot_cfg in enumerate(robot_cfgs):
        robot = env.scene[robot_cfg.name]
        tcp_position = tcp_position_w(robot, robot_cfg.body_ids, env.cfg.tcp_body_offsets[arm])
        vectors.append(
            math_utils.quat_apply_inverse(robot.data.root_quat_w.torch, tail_positions[:, arm] - tcp_position)
        )
    return torch.stack(vectors, dim=1).flatten(start_dim=1)


def tail_tcp_relative_speed(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return free-tail speeds relative to the controlling TCPs [m/s]."""
    _, tail_velocities = tail_state(env, cable_cfgs)
    relative_speeds = []
    for arm, robot_cfg in enumerate(robot_cfgs):
        robot = env.scene[robot_cfg.name]
        tcp_velocity = tcp_velocity_w(robot, robot_cfg.body_ids, env.cfg.tcp_body_offsets[arm])
        relative_speeds.append(torch.linalg.vector_norm(tail_velocities[:, arm] - tcp_velocity, dim=-1))
    return torch.stack(relative_speeds, dim=-1)
