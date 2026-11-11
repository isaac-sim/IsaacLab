# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.events`.

Events write their ``outputs`` for the selected worlds. Ranges are ``(low, high)`` pairs; root ranges are six pairs in
the order x, y, z, roll, pitch, yaw, where MDP1 takes a dict. Each event draws from its use's random stream, the same
draws on both backends.
"""

from __future__ import annotations

import functools

import torch
import warp as wp

from isaaclab.utils import math as math_utils

from ..rng import uniform
from ..terms import Term

# Joint state.


@wp.func
def _reset_joints_by_offset(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    default_joint_vel: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_vel_limits: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    position_range: wp.vec2f,
    velocity_range: wp.vec2f,
    rng: wp.int32,
):
    count = joint_ids.shape[0]
    for k in range(count):
        j = joint_ids[k]
        position = default_joint_pos[world, j] + wp.randf(wp.rand_init(rng, k), position_range[0], position_range[1])
        velocity = default_joint_vel[world, j] + wp.randf(
            wp.rand_init(rng, count + k), velocity_range[0], velocity_range[1]
        )
        limits, velocity_limit = soft_joint_pos_limits[world, j], soft_joint_vel_limits[world, j]
        joint_pos[world, j] = wp.clamp(position, limits[0], limits[1])
        joint_vel[world, j] = wp.clamp(velocity, -velocity_limit, velocity_limit)


@wp.func
def _reset_joints_by_scale(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    default_joint_vel: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_vel_limits: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    position_range: wp.vec2f,
    velocity_range: wp.vec2f,
    rng: wp.int32,
):
    count = joint_ids.shape[0]
    for k in range(count):
        j = joint_ids[k]
        position = default_joint_pos[world, j] * wp.randf(wp.rand_init(rng, k), position_range[0], position_range[1])
        velocity = default_joint_vel[world, j] * wp.randf(
            wp.rand_init(rng, count + k), velocity_range[0], velocity_range[1]
        )
        limits, velocity_limit = soft_joint_pos_limits[world, j], soft_joint_vel_limits[world, j]
        joint_pos[world, j] = wp.clamp(position, limits[0], limits[1])
        joint_vel[world, j] = wp.clamp(velocity, -velocity_limit, velocity_limit)


def _reset_joints_torch(
    mask,
    joint_pos,
    joint_vel,
    default_joint_pos,
    default_joint_vel,
    soft_joint_pos_limits,
    soft_joint_vel_limits,
    joint_ids,
    rng,
    *,
    position_range,
    velocity_range,
    scale: bool,
):
    """Offset or scale the default joint state by uniform samples, clamped to the soft limits."""
    count = len(joint_ids)
    position_samples = uniform(rng, *position_range, (count,))
    velocity_samples = uniform(rng, *velocity_range, (count,), offset=count)
    if scale:
        position = default_joint_pos[:, joint_ids] * position_samples
        velocity = default_joint_vel[:, joint_ids] * velocity_samples
    else:
        position = default_joint_pos[:, joint_ids] + position_samples
        velocity = default_joint_vel[:, joint_ids] + velocity_samples

    limits, velocity_limits = soft_joint_pos_limits[:, joint_ids], soft_joint_vel_limits[:, joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    velocity = velocity.clamp(-velocity_limits, velocity_limits)

    joint_pos[:, joint_ids] = torch.where(mask[:, None], position, joint_pos[:, joint_ids])
    joint_vel[:, joint_ids] = torch.where(mask[:, None], velocity, joint_vel[:, joint_ids])


reset_joints_by_offset = Term(
    torch=functools.partial(_reset_joints_torch, scale=False),
    warp=_reset_joints_by_offset,
    random=True,
)
"""Set the joint state to the default plus uniform offsets from ``position_range`` and ``velocity_range``."""

reset_joints_by_scale = Term(
    torch=functools.partial(_reset_joints_torch, scale=True),
    warp=_reset_joints_by_scale,
    random=True,
)
"""Set the joint state to the default scaled by uniform factors from ``position_range`` and ``velocity_range``."""


# Root state.


@wp.func
def _uniform_spatial(
    rng: wp.int32, offset: int, ranges: wp.types.matrix(shape=(6, 2), dtype=wp.float32)
) -> wp.spatial_vectorf:
    """Six uniform samples, the ``j``-th from ``ranges[j]`` at draw ``offset + j``."""
    sample = wp.spatial_vectorf()
    for j in range(6):
        sample[j] = wp.randf(wp.rand_init(rng, offset + j), ranges[j, 0], ranges[j, 1])
    return sample


@wp.func
def _reset_root_state_uniform(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    pose_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    velocity_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    rng: wp.int32,
):
    pose = _uniform_spatial(rng, 0, pose_range)
    default = default_root_pose[world]
    position = wp.transform_get_translation(default) + env_origins[world] + wp.spatial_top(pose)
    delta = wp.quat_rpy(pose[3], pose[4], pose[5])
    root_link_pose_w[world] = wp.transformf(position, wp.transform_get_rotation(default) * delta)
    root_com_vel_w[world] = default_root_vel[world] + _uniform_spatial(rng, 6, velocity_range)


@wp.func
def _push_by_setting_velocity(
    world: int,
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    velocity_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    rng: wp.int32,
):
    root_com_vel_w[world] = root_com_vel_w[world] + _uniform_spatial(rng, 0, velocity_range)


def _uniform_spatial_torch(rng: torch.Tensor, offset: int, ranges) -> torch.Tensor:
    """Six uniform samples per world, the ``j``-th from ``ranges[j]`` at draw ``offset + j``."""
    return torch.stack([uniform(rng, low, high, offset=offset + j) for j, (low, high) in enumerate(ranges)], dim=-1)


def _reset_root_state_uniform_torch(
    mask,
    root_link_pose_w,
    root_com_vel_w,
    default_root_pose,
    default_root_vel,
    env_origins,
    rng,
    *,
    pose_range,
    velocity_range,
):
    pose = _uniform_spatial_torch(rng, 0, pose_range)
    position = default_root_pose[:, :3] + env_origins + pose[:, :3]
    delta = math_utils.quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    orientation = math_utils.quat_mul(default_root_pose[:, 3:7], delta)
    velocity = default_root_vel + _uniform_spatial_torch(rng, 6, velocity_range)

    root_link_pose_w.copy_(torch.where(mask[:, None], torch.cat([position, orientation], -1), root_link_pose_w))
    root_com_vel_w.copy_(torch.where(mask[:, None], velocity, root_com_vel_w))


def _push_by_setting_velocity_torch(mask, root_com_vel_w, rng, *, velocity_range):
    velocity = root_com_vel_w + _uniform_spatial_torch(rng, 0, velocity_range)
    root_com_vel_w.copy_(torch.where(mask[:, None], velocity, root_com_vel_w))


reset_root_state_uniform = Term(torch=_reset_root_state_uniform_torch, warp=_reset_root_state_uniform, random=True)
"""Set the root pose to the default, at the environment origin, offset by uniform samples from ``pose_range``,
and the root velocity to the default plus uniform samples from ``velocity_range``. Floating bases only."""

push_by_setting_velocity = Term(torch=_push_by_setting_velocity_torch, warp=_push_by_setting_velocity, random=True)
"""Add uniform samples from ``velocity_range`` to the root velocity. Floating bases only."""
