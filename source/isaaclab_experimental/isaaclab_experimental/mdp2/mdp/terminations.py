# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.terminations`.

Each term returns whether to end each world's episode. Time-outs are :class:`~isaaclab_experimental.mdp2.TimeLimit`.
"""

from __future__ import annotations

import warp as wp

from ..terms import Term
from .observations import projected_gravity_b, projected_gravity_torch
from .rewards import contact_count, contact_mask_torch

# Episode.


@wp.func
def _is_positive(world: int, value: wp.array(dtype=float)) -> bool:
    return value[world] > 0.0


def _is_positive_torch(value):
    return value > 0.0


is_positive = Term(torch=_is_positive_torch, warp=_is_positive)
"""Whether the ``value`` input, one float per world such as a quantity, is positive."""


# Joint state.


@wp.func
def _joint_pos_out_of_manual_limit(
    world: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32), bounds: wp.vec2f
) -> bool:
    outside = bool(False)
    for k in range(joint_ids.shape[0]):
        position = joint_pos[world, joint_ids[k]]
        outside = outside or position < bounds[0] or position > bounds[1]
    return outside


def _joint_pos_out_of_manual_limit_torch(joint_pos, joint_ids, *, bounds):
    position = joint_pos[:, joint_ids]
    return ((position < bounds[0]) | (position > bounds[1])).any(-1)


joint_pos_out_of_manual_limit = Term(torch=_joint_pos_out_of_manual_limit_torch, warp=_joint_pos_out_of_manual_limit)
"""Whether a joint position is outside ``bounds``, a ``(lower, upper)`` pair."""


@wp.func
def _joint_vel_out_of_limit(
    world: int,
    joint_vel: wp.array2d(dtype=float),
    soft_joint_vel_limits: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> bool:
    outside = bool(False)
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        outside = outside or wp.abs(joint_vel[world, j]) > soft_joint_vel_limits[world, j]
    return outside


def _joint_vel_out_of_limit_torch(joint_vel, soft_joint_vel_limits, joint_ids):
    return (joint_vel[:, joint_ids].abs() > soft_joint_vel_limits[:, joint_ids]).any(-1)


joint_vel_out_of_limit = Term(torch=_joint_vel_out_of_limit_torch, warp=_joint_vel_out_of_limit)
"""Whether a joint speed exceeds its soft velocity limit."""


# Root state.


@wp.func
def _root_height_below_minimum(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), minimum_height: float
) -> bool:
    return wp.transform_get_translation(root_link_pose_w[world])[2] < minimum_height


root_height_below_minimum = Term(
    torch=lambda root_link_pose_w, *, minimum_height: root_link_pose_w[:, 2] < minimum_height,
    warp=_root_height_below_minimum,
)
"""Whether the root is lower than ``minimum_height`` in the world frame."""


@wp.func
def _bad_orientation(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    gravity_w: wp.array(dtype=wp.vec3f),
    limit_angle: float,
) -> bool:
    gravity = projected_gravity_b(world, root_link_pose_w, gravity_w)
    return wp.abs(wp.acos(-gravity[2])) > limit_angle


def _bad_orientation_torch(root_link_pose_w, gravity_w, *, limit_angle):
    gravity = projected_gravity_torch(root_link_pose_w, gravity_w)
    return gravity[:, 2].neg().acos().abs() > limit_angle


bad_orientation = Term(torch=_bad_orientation_torch, warp=_bad_orientation)
"""Whether the root is tilted more than ``limit_angle`` radians from upright."""


# Contacts.


@wp.func
def _illegal_contact(
    world: int,
    net_normal_forces_w_history: wp.array3d(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> bool:
    return contact_count(world, net_normal_forces_w_history, body_ids, threshold) > 0


def _illegal_contact_torch(net_normal_forces_w_history, body_ids, *, threshold):
    return contact_mask_torch(net_normal_forces_w_history, body_ids, threshold).any(-1)


illegal_contact = Term(torch=_illegal_contact_torch, warp=_illegal_contact)
"""Whether a normal contact force on the ``body_ids`` bodies exceeded ``threshold`` in the sensor history."""
