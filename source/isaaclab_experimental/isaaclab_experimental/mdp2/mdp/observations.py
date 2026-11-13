# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.observations`.

Joint terms read the joints of a subset input, ``joint_ids=robot.joints(".*")``, and output one value per joint.
Root terms read the root state of an asset, ``root_link_pose_w`` and ``root_com_vel_w``.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils import math as math_utils

from ..terms import Term

# Joint state.


@wp.func
def _joint_pos(
    world: int, joint: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)
) -> float:
    return joint_pos[world, joint_ids[joint]]


joint_pos = Term(
    torch=lambda joint_pos, joint_ids: joint_pos[:, joint_ids],
    warp=_joint_pos,
    shape=("worlds", "joint_ids"),
)
"""Joint positions."""


@wp.func
def _joint_pos_rel(
    world: int,
    joint: int,
    joint_pos: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    j = joint_ids[joint]
    return joint_pos[world, j] - default_joint_pos[world, j]


joint_pos_rel = Term(
    torch=lambda joint_pos, default_joint_pos, joint_ids: joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids],
    warp=_joint_pos_rel,
    shape=("worlds", "joint_ids"),
)
"""Joint positions relative to the default joint positions."""


@wp.func
def _joint_vel(
    world: int, joint: int, joint_vel: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)
) -> float:
    return joint_vel[world, joint_ids[joint]]


joint_vel = Term(
    torch=lambda joint_vel, joint_ids: joint_vel[:, joint_ids],
    warp=_joint_vel,
    shape=("worlds", "joint_ids"),
)
"""Joint velocities."""


@wp.func
def _joint_vel_rel(
    world: int,
    joint: int,
    joint_vel: wp.array2d(dtype=float),
    default_joint_vel: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    j = joint_ids[joint]
    return joint_vel[world, j] - default_joint_vel[world, j]


joint_vel_rel = Term(
    torch=lambda joint_vel, default_joint_vel, joint_ids: joint_vel[:, joint_ids] - default_joint_vel[:, joint_ids],
    warp=_joint_vel_rel,
    shape=("worlds", "joint_ids"),
)
"""Joint velocities relative to the default joint velocities."""


@wp.func
def _joint_pos_limit_normalized(
    world: int,
    joint: int,
    joint_pos: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    j = joint_ids[joint]
    lower = soft_joint_pos_limits[world, j][0]
    upper = soft_joint_pos_limits[world, j][1]
    return 2.0 * (joint_pos[world, j] - 0.5 * (lower + upper)) / (upper - lower)


def _joint_pos_limit_normalized_torch(joint_pos, soft_joint_pos_limits, joint_ids):
    limits = soft_joint_pos_limits[:, joint_ids]
    return math_utils.scale_transform(joint_pos[:, joint_ids], limits[..., 0], limits[..., 1])


joint_pos_limit_normalized = Term(
    torch=_joint_pos_limit_normalized_torch,
    warp=_joint_pos_limit_normalized,
    shape=("worlds", "joint_ids"),
)
"""Joint positions scaled to [-1, 1] between the soft joint position limits."""


# Root state.


@wp.func
def _root_pos_w(
    world: int, axis: int, root_link_pose_w: wp.array(dtype=wp.transformf), env_origins: wp.array(dtype=wp.vec3f)
) -> float:
    return wp.transform_get_translation(root_link_pose_w[world])[axis] - env_origins[world][axis]


root_pos_w = Term(
    torch=lambda root_link_pose_w, env_origins: root_link_pose_w[:, :3] - env_origins,
    warp=_root_pos_w,
    shape=("worlds", 3),
)
"""Root position relative to the world's environment origin."""


@wp.func
def _root_quat_w(
    world: int, element: int, root_link_pose_w: wp.array(dtype=wp.transformf), make_quat_unique: bool
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    if make_quat_unique and quat[3] < 0.0:
        quat = -quat
    return quat[element]


def _root_quat_w_torch(root_link_pose_w, *, make_quat_unique):
    quat = root_link_pose_w[:, 3:7]
    return math_utils.quat_unique(quat) if make_quat_unique else quat


root_quat_w = Term(torch=_root_quat_w_torch, warp=_root_quat_w, shape=("worlds", 4))
"""Root orientation ``(x, y, z, w)``, with a non-negative real part if ``make_quat_unique``."""


@wp.func
def _base_lin_vel(
    world: int,
    axis: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    return wp.quat_rotate_inv(quat, wp.spatial_top(root_com_vel_w[world]))[axis]


def _base_lin_vel_torch(root_link_pose_w, root_com_vel_w):
    return math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], root_com_vel_w[:, :3])


base_lin_vel = Term(torch=_base_lin_vel_torch, warp=_base_lin_vel, shape=("worlds", 3))
"""Root linear velocity in the root frame."""


@wp.func
def _base_ang_vel(
    world: int,
    axis: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    return wp.quat_rotate_inv(quat, wp.spatial_bottom(root_com_vel_w[world]))[axis]


def _base_ang_vel_torch(root_link_pose_w, root_com_vel_w):
    return math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], root_com_vel_w[:, 3:6])


base_ang_vel = Term(torch=_base_ang_vel_torch, warp=_base_ang_vel, shape=("worlds", 3))
"""Root angular velocity in the root frame."""


@wp.func
def projected_gravity_b(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), gravity_w: wp.array(dtype=wp.vec3f)
) -> wp.vec3f:
    """Return the direction of gravity in the root frame of a world."""
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    return wp.quat_rotate_inv(quat, wp.normalize(gravity_w[world]))


def projected_gravity_torch(root_link_pose_w: torch.Tensor, gravity_w: torch.Tensor) -> torch.Tensor:
    """Return the direction of gravity in the root frame, ``(num_worlds, 3)``."""
    return math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], math_utils.normalize(gravity_w))


@wp.func
def _projected_gravity(
    world: int, axis: int, root_link_pose_w: wp.array(dtype=wp.transformf), gravity_w: wp.array(dtype=wp.vec3f)
) -> float:
    return projected_gravity_b(world, root_link_pose_w, gravity_w)[axis]


projected_gravity = Term(torch=projected_gravity_torch, warp=_projected_gravity, shape=("worlds", 3))
"""Direction of gravity in the root frame."""


# Actions.


@wp.func
def _last_action(world: int, action: int, actions: wp.array2d(dtype=float)) -> float:
    return actions[world, action]


last_action = Term(torch=lambda actions: actions, warp=_last_action, shape=("worlds", "actions"))
"""The last actions, bound to ``mdp2.last_action``."""


# Commands.


@wp.func
def _generated_commands(world: int, element: int, command: wp.array2d(dtype=float)) -> float:
    return command[world, element]


generated_commands = Term(torch=lambda command: command, warp=_generated_commands, shape=("worlds", "command"))
"""The value of a command, bound to ``mdp2.commands.<name>``."""
