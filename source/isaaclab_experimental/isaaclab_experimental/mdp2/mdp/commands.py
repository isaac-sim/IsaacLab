# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms, the MDP2 counterparts of the command terms of :mod:`isaaclab.envs.mdp.commands`.

A command use samples its term when its resampling timer expires and on reset; observe it with
:data:`~isaaclab_experimental.mdp2.mdp.generated_commands`. Each command draws from its use's random stream, the same
draws on both backends.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils import math as math_utils

from ..rng import uniform
from ..terms import Term

# Velocity.


@wp.func
def _uniform_velocity_command(
    world: int,
    axis: int,
    lin_vel_x: wp.vec2f,
    lin_vel_y: wp.vec2f,
    ang_vel_z: wp.vec2f,
    rel_standing_envs: float,
    rng: wp.int32,
) -> float:
    if wp.randf(wp.rand_init(rng, 3)) <= rel_standing_envs:
        return 0.0
    bounds = lin_vel_x
    if axis == 1:
        bounds = lin_vel_y
    elif axis == 2:
        bounds = ang_vel_z
    return wp.randf(wp.rand_init(rng, axis), bounds[0], bounds[1])


def _uniform_velocity_command_torch(rng, *, lin_vel_x, lin_vel_y, ang_vel_z, rel_standing_envs):
    velocity = torch.stack(
        [uniform(rng, *bounds, offset=axis) for axis, bounds in enumerate((lin_vel_x, lin_vel_y, ang_vel_z))], dim=-1
    )
    standing = uniform(rng, 0.0, 1.0, offset=3) <= rel_standing_envs
    return torch.where(standing[:, None], torch.zeros_like(velocity), velocity)


uniform_velocity_command = Term(
    torch=_uniform_velocity_command_torch, warp=_uniform_velocity_command, shape=("worlds", 3), random=True
)
"""Base velocity ``(lin_vel_x, lin_vel_y, ang_vel_z)`` drawn uniformly from the given ranges, or zero for a fraction
``rel_standing_envs`` of the samples. Without heading control, unlike MDP1's ``UniformVelocityCommand`` with
``heading_command``."""


# Pose.


@wp.func
def _uniform_pose_command(
    world: int,
    element: int,
    pose_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    make_quat_unique: bool,
    rng: wp.int32,
) -> float:
    if element < 3:
        return wp.randf(wp.rand_init(rng, element), pose_range[element, 0], pose_range[element, 1])

    roll = wp.randf(wp.rand_init(rng, 3), pose_range[3, 0], pose_range[3, 1])
    pitch = wp.randf(wp.rand_init(rng, 4), pose_range[4, 0], pose_range[4, 1])
    yaw = wp.randf(wp.rand_init(rng, 5), pose_range[5, 0], pose_range[5, 1])
    quat = wp.quat_rpy(roll, pitch, yaw)
    if make_quat_unique and quat[3] < 0.0:
        quat = -quat
    return quat[element - 3]


def _uniform_pose_command_torch(rng, *, pose_range, make_quat_unique):
    samples = [uniform(rng, low, high, offset=j) for j, (low, high) in enumerate(pose_range)]
    quat = math_utils.quat_from_euler_xyz(*samples[3:])
    if make_quat_unique:
        quat = math_utils.quat_unique(quat)
    return torch.cat([torch.stack(samples[:3], dim=-1), quat], dim=-1)


uniform_pose_command = Term(
    torch=_uniform_pose_command_torch, warp=_uniform_pose_command, shape=("worlds", 7), random=True
)
"""Pose ``(x, y, z, qx, qy, qz, qw)`` in the robot's base frame, drawn uniformly from ``pose_range``, six ``(low,
high)`` pairs in the order x, y, z, roll, pitch, yaw. With ``make_quat_unique``, the real part is non-negative."""
