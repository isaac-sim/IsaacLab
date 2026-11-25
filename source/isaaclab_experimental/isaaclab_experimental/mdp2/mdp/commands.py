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
from ..terms import Implementation, State, Term

# Velocity.


@wp.func
def _write_velocity(
    world: int,
    command: wp.array2d(dtype=float),
    lin_vel_x: wp.vec2f,
    lin_vel_y: wp.vec2f,
    ang_vel_z: wp.vec2f,
    rel_standing_envs: float,
    rng: wp.int32,
) -> bool:
    """Write a uniform velocity sample to a world's command, zero if the world stands, and return whether it stands."""
    standing = wp.randf(wp.rand_init(rng, 3)) <= rel_standing_envs
    velocity = wp.vec3f(
        wp.randf(wp.rand_init(rng, 0), lin_vel_x[0], lin_vel_x[1]),
        wp.randf(wp.rand_init(rng, 1), lin_vel_y[0], lin_vel_y[1]),
        wp.randf(wp.rand_init(rng, 2), ang_vel_z[0], ang_vel_z[1]),
    )
    if standing:
        velocity = wp.vec3f(0.0)

    for axis in range(3):
        command[world, axis] = velocity[axis]
    return standing


def _velocity_sample_torch(
    rng: torch.Tensor,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    rel_standing_envs: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a uniform velocity sample per world, zero for standing worlds, and whether each world stands."""
    ranges = (lin_vel_x, lin_vel_y, ang_vel_z)
    velocity = torch.stack([uniform(rng, *bounds, offset=axis) for axis, bounds in enumerate(ranges)], dim=-1)
    standing = uniform(rng, 0.0, 1.0, offset=3) <= rel_standing_envs
    return torch.where(standing[:, None], torch.zeros_like(velocity), velocity), standing


@wp.func
def _uniform_velocity_command(
    world: int,
    command: wp.array2d(dtype=float),
    lin_vel_x: wp.vec2f,
    lin_vel_y: wp.vec2f,
    ang_vel_z: wp.vec2f,
    rel_standing_envs: float,
    rng: wp.int32,
):
    _write_velocity(world, command, lin_vel_x, lin_vel_y, ang_vel_z, rel_standing_envs, rng)


def _uniform_velocity_command_torch(
    mask: torch.Tensor,
    command: torch.Tensor,
    rng: torch.Tensor,
    *,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    rel_standing_envs: float,
) -> None:
    velocity, _ = _velocity_sample_torch(rng, lin_vel_x, lin_vel_y, ang_vel_z, rel_standing_envs)
    command.copy_(torch.where(mask[:, None], velocity, command))


uniform_velocity_command = Term(
    torch=_uniform_velocity_command_torch, warp=_uniform_velocity_command, shape=("worlds", 3), random=True
)
"""Base velocity ``(lin_vel_x, lin_vel_y, ang_vel_z)`` drawn uniformly from the given ranges.

The velocity is zero for a fraction ``rel_standing_envs`` of the samples. For heading control, use
:data:`heading_velocity_command`.
"""


@wp.func
def _heading_velocity_command(
    world: int,
    command: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    lin_vel_x: wp.vec2f,
    lin_vel_y: wp.vec2f,
    ang_vel_z: wp.vec2f,
    heading: wp.vec2f,
    rel_standing_envs: float,
    rel_heading_envs: float,
    rng: wp.int32,
):
    standing = _write_velocity(world, command, lin_vel_x, lin_vel_y, ang_vel_z, rel_standing_envs, rng)

    target = wp.randf(wp.rand_init(rng, 5), heading[0], heading[1])
    if standing or wp.randf(wp.rand_init(rng, 4)) > rel_heading_envs:
        target = wp.nan
    heading_target[world] = target


def _heading_velocity_command_torch(
    mask: torch.Tensor,
    command: torch.Tensor,
    heading_target: torch.Tensor,
    rng: torch.Tensor,
    *,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
    rel_standing_envs: float,
    rel_heading_envs: float,
) -> None:
    velocity, standing = _velocity_sample_torch(rng, lin_vel_x, lin_vel_y, ang_vel_z, rel_standing_envs)
    controlled = ~standing & (uniform(rng, 0.0, 1.0, offset=4) <= rel_heading_envs)
    target = torch.where(controlled, uniform(rng, *heading, offset=5), torch.nan)

    command.copy_(torch.where(mask[:, None], velocity, command))
    heading_target.copy_(torch.where(mask, target, heading_target))


@wp.func
def _steer_to_heading(
    world: int,
    command: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    ang_vel_z: wp.vec2f,
    heading_control_stiffness: float,
):
    target = heading_target[world]
    if wp.isnan(target):
        return

    quat = wp.transform_get_rotation(root_link_pose_w[world])
    forward = wp.quat_rotate(quat, wp.vec3f(1.0, 0.0, 0.0))
    error = target - wp.atan2(forward[1], forward[0])
    error = wp.atan2(wp.sin(error), wp.cos(error))
    command[world, 2] = wp.clamp(heading_control_stiffness * error, ang_vel_z[0], ang_vel_z[1])


def _steer_to_heading_torch(
    command: torch.Tensor,
    heading_target: torch.Tensor,
    root_link_pose_w: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    ang_vel_z: tuple[float, float],
    heading_control_stiffness: float,
) -> None:
    # The heading is the direction of the root's x axis in the ground plane.
    x, y, z, w = root_link_pose_w[:, 3:7].unbind(-1)
    heading = torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    error = math_utils.wrap_to_pi(heading_target - heading)
    angular = (heading_control_stiffness * error).clip(*ang_vel_z)
    # The mask, if any, holds the worlds of the command's world config.
    steered = ~heading_target.isnan() if mask is None else mask & ~heading_target.isnan()
    command[:, 2] = torch.where(steered, angular, command[:, 2])


heading_velocity_command = Term(
    torch=Implementation(compute=_heading_velocity_command_torch, update=_steer_to_heading_torch),
    warp=Implementation(compute=_heading_velocity_command, update=_steer_to_heading),
    shape=("worlds", 3),
    state={"heading_target": State(initial=float("nan"))},
    random=True,
)
"""Base velocity like :data:`uniform_velocity_command`, with the yaw rate steered to a target heading.

Each sample also draws a target heading from ``heading`` for a fraction ``rel_heading_envs`` of the moving samples.
Every step, their ``ang_vel_z`` steers the root's yaw towards it with gain ``heading_control_stiffness``, clipped to
``ang_vel_z``, as MDP1's ``UniformVelocityCommand`` with ``heading_command``. Bind ``root_link_pose_w`` to the robot.
"""


@wp.func
def _heading_velocity_command_steered_on_resample(
    world: int,
    command: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    lin_vel_x: wp.vec2f,
    lin_vel_y: wp.vec2f,
    ang_vel_z: wp.vec2f,
    heading: wp.vec2f,
    rel_standing_envs: float,
    rel_heading_envs: float,
    heading_control_stiffness: float,
    rng: wp.int32,
):
    _heading_velocity_command(
        world,
        command,
        heading_target,
        lin_vel_x,
        lin_vel_y,
        ang_vel_z,
        heading,
        rel_standing_envs,
        rel_heading_envs,
        rng,
    )
    _steer_to_heading(world, command, heading_target, root_link_pose_w, ang_vel_z, heading_control_stiffness)


def _heading_velocity_command_steered_on_resample_torch(
    mask: torch.Tensor,
    command: torch.Tensor,
    heading_target: torch.Tensor,
    root_link_pose_w: torch.Tensor,
    rng: torch.Tensor,
    *,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
    rel_standing_envs: float,
    rel_heading_envs: float,
    heading_control_stiffness: float,
) -> None:
    _heading_velocity_command_torch(
        mask,
        command,
        heading_target,
        rng,
        lin_vel_x=lin_vel_x,
        lin_vel_y=lin_vel_y,
        ang_vel_z=ang_vel_z,
        heading=heading,
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=rel_heading_envs,
    )
    _steer_to_heading_torch(
        command,
        heading_target,
        root_link_pose_w,
        mask,
        ang_vel_z=ang_vel_z,
        heading_control_stiffness=heading_control_stiffness,
    )


heading_velocity_command_steered_on_resample = Term(
    torch=Implementation(compute=_heading_velocity_command_steered_on_resample_torch, update=_steer_to_heading_torch),
    warp=Implementation(compute=_heading_velocity_command_steered_on_resample, update=_steer_to_heading),
    shape=("worlds", 3),
    state={"heading_target": State(initial=float("nan"))},
    random=True,
)
"""Like :data:`heading_velocity_command`, but a resample also steers the yaw rate to the new target at once.

The first observation after a reset then carries the steered yaw rate of the reset pose, as MDP1's command manager
updates the commands after a reset, instead of the drawn ``ang_vel_z``. Bind ``root_link_pose_w`` to the robot.
"""


# Pose.


@wp.func
def _uniform_pose_command(
    world: int,
    command: wp.array2d(dtype=float),
    pose_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    make_quat_unique: bool,
    rng: wp.int32,
):
    for axis in range(3):
        command[world, axis] = wp.randf(wp.rand_init(rng, axis), pose_range[axis, 0], pose_range[axis, 1])

    roll = wp.randf(wp.rand_init(rng, 3), pose_range[3, 0], pose_range[3, 1])
    pitch = wp.randf(wp.rand_init(rng, 4), pose_range[4, 0], pose_range[4, 1])
    yaw = wp.randf(wp.rand_init(rng, 5), pose_range[5, 0], pose_range[5, 1])
    quat = wp.quat_rpy(roll, pitch, yaw)
    if make_quat_unique and quat[3] < 0.0:
        quat = -quat
    for k in range(4):
        command[world, 3 + k] = quat[k]


def _uniform_pose_command_torch(
    mask: torch.Tensor,
    command: torch.Tensor,
    rng: torch.Tensor,
    *,
    pose_range: tuple[tuple[float, float], ...],
    make_quat_unique: bool,
) -> None:
    samples = [uniform(rng, low, high, offset=j) for j, (low, high) in enumerate(pose_range)]
    quat = math_utils.quat_from_euler_xyz(*samples[3:])
    if make_quat_unique:
        quat = math_utils.quat_unique(quat)
    pose = torch.cat([torch.stack(samples[:3], dim=-1), quat], dim=-1)
    command.copy_(torch.where(mask[:, None], pose, command))


uniform_pose_command = Term(
    torch=_uniform_pose_command_torch, warp=_uniform_pose_command, shape=("worlds", 7), random=True
)
"""Pose ``(x, y, z, qx, qy, qz, qw)`` in the robot's base frame, drawn uniformly from ``pose_range``.

``pose_range`` holds six ``(low, high)`` pairs in the order x, y, z, roll, pitch, yaw. With ``make_quat_unique``, the
real part is non-negative.
"""
