# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.rewards`.

Each term returns one value per world, which the reward use weighs. Joint terms sum over the joints of a subset
input, ``joint_ids=robot.joints(".*")``.
"""

from __future__ import annotations

import warp as wp

from isaaclab.utils import math as math_utils

from ..terms import Implementation, State, Term
from .observations import projected_gravity_b, projected_gravity_torch

FIRST_CONTACT_TOLERANCE = wp.constant(1.0e-4)
"""Margin [s] on the step duration within which a contact counts as new: below any physics step, above rounding."""


# Episode.


@wp.func
def _is_alive(world: int, terminated: wp.array(dtype=wp.bool)) -> float:
    return wp.where(terminated[world], 0.0, 1.0)


is_alive = Term(torch=lambda terminated: (~terminated).float(), warp=_is_alive)
"""One for worlds whose episode did not terminate this step, bound to ``mdp2.terminated``."""


@wp.func
def _is_terminated(world: int, terminated: wp.array(dtype=wp.bool)) -> float:
    return wp.where(terminated[world], 1.0, 0.0)


is_terminated = Term(torch=lambda terminated: terminated.float(), warp=_is_terminated)
"""One for worlds whose episode terminated this step, not by a time-out, bound to ``mdp2.terminated``."""


# Joint state.


@wp.func
def _joint_vel_l1(world: int, joint_vel: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        total += wp.abs(joint_vel[world, joint_ids[k]])
    return total


joint_vel_l1 = Term(torch=lambda joint_vel, joint_ids: joint_vel[:, joint_ids].abs().sum(-1), warp=_joint_vel_l1)
"""Sum of the absolute joint velocities."""


@wp.func
def _joint_vel_l2(world: int, joint_vel: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        velocity = joint_vel[world, joint_ids[k]]
        total += velocity * velocity
    return total


joint_vel_l2 = Term(torch=lambda joint_vel, joint_ids: joint_vel[:, joint_ids].square().sum(-1), warp=_joint_vel_l2)
"""Sum of the squared joint velocities."""


@wp.func
def _joint_deviation_l1(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        total += wp.abs(joint_pos[world, j] - default_joint_pos[world, j])
    return total


def _joint_deviation_l1_torch(joint_pos, default_joint_pos, joint_ids):
    deviation = joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]
    return deviation.abs().sum(-1)


joint_deviation_l1 = Term(torch=_joint_deviation_l1_torch, warp=_joint_deviation_l1)
"""Sum of the absolute deviations of the joint positions from their defaults."""


@wp.func
def _joint_pos_target_l2(
    world: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32), target: float
) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        error = joint_pos[world, joint_ids[k]] - target
        total += error * error
    return total


joint_pos_target_l2 = Term(
    torch=lambda joint_pos, joint_ids, *, target: (joint_pos[:, joint_ids] - target).square().sum(-1),
    warp=_joint_pos_target_l2,
)
"""Sum of the squared errors of the joint positions from ``target``."""


@wp.func
def _joint_pos_limits(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        position = joint_pos[world, j]
        limits = soft_joint_pos_limits[world, j]
        total += wp.max(limits[0] - position, 0.0) + wp.max(position - limits[1], 0.0)
    return total


def _joint_pos_limits_torch(joint_pos, soft_joint_pos_limits, joint_ids):
    position = joint_pos[:, joint_ids]
    limits = soft_joint_pos_limits[:, joint_ids]

    below = (limits[..., 0] - position).clip(min=0.0)
    above = (position - limits[..., 1]).clip(min=0.0)
    return (below + above).sum(-1)


joint_pos_limits = Term(torch=_joint_pos_limits_torch, warp=_joint_pos_limits)
"""Sum of the distances of the joint positions beyond their soft limits."""


# Root state.


@wp.func
def _lin_vel_z_l2(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), root_com_vel_w: wp.array(dtype=wp.spatial_vectorf)
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    velocity = wp.quat_rotate_inv(quat, wp.spatial_top(root_com_vel_w[world]))
    return velocity[2] * velocity[2]


def _lin_vel_z_l2_torch(root_link_pose_w, root_com_vel_w):
    velocity = math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], root_com_vel_w[:, :3])
    return velocity[:, 2].square()


lin_vel_z_l2 = Term(torch=_lin_vel_z_l2_torch, warp=_lin_vel_z_l2)
"""Squared vertical velocity of the root, in the root frame."""


@wp.func
def _ang_vel_xy_l2(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), root_com_vel_w: wp.array(dtype=wp.spatial_vectorf)
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    velocity = wp.quat_rotate_inv(quat, wp.spatial_bottom(root_com_vel_w[world]))
    return velocity[0] * velocity[0] + velocity[1] * velocity[1]


def _ang_vel_xy_l2_torch(root_link_pose_w, root_com_vel_w):
    velocity = math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], root_com_vel_w[:, 3:6])
    return velocity[:, :2].square().sum(-1)


ang_vel_xy_l2 = Term(torch=_ang_vel_xy_l2_torch, warp=_ang_vel_xy_l2)
"""Squared roll and pitch rates of the root, in the root frame."""


@wp.func
def _flat_orientation_l2(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), gravity_w: wp.array(dtype=wp.vec3f)
) -> float:
    gravity = projected_gravity_b(world, root_link_pose_w, gravity_w)
    return gravity[0] * gravity[0] + gravity[1] * gravity[1]


def _flat_orientation_l2_torch(root_link_pose_w, gravity_w):
    gravity = projected_gravity_torch(root_link_pose_w, gravity_w)
    return gravity[:, :2].square().sum(-1)


flat_orientation_l2 = Term(torch=_flat_orientation_l2_torch, warp=_flat_orientation_l2)
"""Squared horizontal components of the gravity direction in the root frame: zero when the root is level."""


# Actions.


@wp.func
def _action_l2(world: int, actions: wp.array2d(dtype=float)) -> float:
    total = float(0.0)
    for k in range(actions.shape[1]):
        total += actions[world, k] * actions[world, k]
    return total


action_l2 = Term(torch=lambda actions: actions.square().sum(-1), warp=_action_l2)
"""Sum of the squared actions, bound to ``mdp2.last_action``."""


@wp.func
def _action_rate_l2(world: int, actions: wp.array2d(dtype=float), previous_actions: wp.array2d(dtype=float)) -> float:
    total = float(0.0)
    for k in range(actions.shape[1]):
        change = actions[world, k] - previous_actions[world, k]
        total += change * change
    return total


@wp.func
def _action_rate_update(world: int, actions: wp.array2d(dtype=float), previous_actions: wp.array2d(dtype=float)):
    for k in range(actions.shape[1]):
        previous_actions[world, k] = actions[world, k]


action_rate_l2 = Term(
    torch=Implementation(
        compute=lambda actions, previous_actions: (actions - previous_actions).square().sum(-1),
        update=lambda actions, previous_actions: previous_actions.copy_(actions),
    ),
    warp=Implementation(compute=_action_rate_l2, update=_action_rate_update),
    state={"previous_actions": State(("worlds", "actions"))},
)
"""Sum of the squared changes of the actions since the last step, bound to ``mdp2.last_action``.

The previous actions are zero after a reset, as in MDP1.
"""


# Contacts.


@wp.func
def contact_count(
    world: int,
    net_normal_forces_w_history: wp.array3d(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> int:
    """Return the number of bodies whose normal contact force exceeded ``threshold`` in the sensor history."""
    count = int(0)
    for k in range(body_ids.shape[0]):
        peak = float(0.0)
        for step in range(net_normal_forces_w_history.shape[1]):
            peak = wp.max(peak, wp.length(net_normal_forces_w_history[world, step, body_ids[k]]))
        if peak > threshold:
            count += 1
    return count


def contact_mask_torch(net_normal_forces_w_history, body_ids, threshold):
    """Return whether each body's normal contact force exceeded ``threshold`` in the sensor history."""
    return net_normal_forces_w_history[:, :, body_ids].norm(dim=-1).amax(dim=1) > threshold


@wp.func
def _undesired_contacts(
    world: int,
    net_normal_forces_w_history: wp.array3d(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> float:
    return float(contact_count(world, net_normal_forces_w_history, body_ids, threshold))


def _undesired_contact_mask_torch(net_normal_forces_w_history, body_ids, *, threshold):
    return contact_mask_torch(net_normal_forces_w_history, body_ids, threshold).sum(-1).float()


undesired_contacts = Term(torch=_undesired_contact_mask_torch, warp=_undesired_contacts)
"""Number of the ``body_ids`` bodies whose normal contact force exceeded ``threshold`` in the sensor history."""


@wp.func
def _feet_air_time(
    world: int,
    current_contact_time: wp.array2d(dtype=float),
    last_air_time: wp.array2d(dtype=float),
    command: wp.array2d(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
    step_dt: float,
) -> float:
    if wp.length(wp.vec2f(command[world, 0], command[world, 1])) <= 0.1:
        return 0.0

    total = float(0.0)
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        contact_time = current_contact_time[world, body]
        if contact_time > 0.0 and contact_time < step_dt + FIRST_CONTACT_TOLERANCE:
            total += last_air_time[world, body] - threshold
    return total


def _feet_air_time_torch(current_contact_time, last_air_time, command, body_ids, *, threshold, step_dt):
    contact_time = current_contact_time[:, body_ids]
    first_contact = (contact_time > 0.0) & (contact_time < step_dt + FIRST_CONTACT_TOLERANCE)

    reward = ((last_air_time[:, body_ids] - threshold) * first_contact).sum(-1)
    return reward * (command[:, :2].norm(dim=-1) > 0.1)


feet_air_time = Term(torch=_feet_air_time_torch, warp=_feet_air_time)
"""Sum over the ``body_ids`` feet that touched down this step of their air time minus ``threshold``.

Zero for worlds whose velocity ``command`` is at most 0.1. Bind ``step_dt`` to ``mdp2.step_dt``.
"""
