# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..shoelace_constants import TCP_OFFSET
from .utils import tail_state


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


def gripper_close_error(
    env: ManagerBasedEnv,
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return positive actual-minus-commanded gripper position residuals [m]."""
    errors = []
    for robot_cfg in robot_cfgs:
        robot = env.scene[robot_cfg.name]
        actual = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
        target = robot.data.joint_pos_target.torch[:, robot_cfg.joint_ids]
        errors.append((actual - target).clamp_min_(0.0))
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
        hand_position = robot.data.body_pos_w.torch[:, robot_cfg.body_ids[0]]
        hand_quaternion = robot.data.body_quat_w.torch[:, robot_cfg.body_ids[0]]
        tcp_position = hand_position + _tcp_offset_w(hand_quaternion)
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
        hand_quaternion = robot.data.body_quat_w.torch[:, robot_cfg.body_ids[0]]
        hand_velocity = robot.data.body_link_vel_w.torch[:, robot_cfg.body_ids[0]]
        tcp_offset_w = _tcp_offset_w(hand_quaternion)
        tcp_velocity = hand_velocity[:, :3] + torch.linalg.cross(hand_velocity[:, 3:], tcp_offset_w, dim=-1)
        relative_speeds.append(torch.linalg.vector_norm(tail_velocities[:, arm] - tcp_velocity, dim=-1))
    return torch.stack(relative_speeds, dim=-1)


def _tcp_offset_w(hand_quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate the hand-local TCP offset into world coordinates [m], shape [N, 3]."""
    offset = hand_quaternion.new_tensor(TCP_OFFSET).expand(hand_quaternion.shape[0], -1)
    return math_utils.quat_apply(hand_quaternion, offset)
