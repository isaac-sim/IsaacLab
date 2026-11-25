# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.rewards`.

Each term returns one value per world, which the reward use weighs. Joint terms sum over the joints of a subset
input, ``joint_ids=robot.joints(".*")``.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils import math as math_utils
from isaaclab.utils.warp.math import quat_error_magnitude, yaw_quat

from ..eager import TermContext
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


@wp.func
def _is_terminated_before_time_out(
    world: int, terminated: wp.array(dtype=wp.bool), time_out: wp.array(dtype=wp.bool)
) -> float:
    return float(terminated[world] and not time_out[world])


is_terminated_before_time_out = Term(
    torch=lambda terminated, time_out: (terminated & ~time_out).float(), warp=_is_terminated_before_time_out
)
"""One for worlds whose episode terminated this step without a time-out, bound to ``mdp2.terminated`` and
``mdp2.time_out``."""


@wp.func
def _value(world: int, value: wp.array(dtype=float)) -> float:
    return value[world]


def _value_torch(value: torch.Tensor) -> torch.Tensor:
    return value


value = Term(torch=_value_torch, warp=_value)
"""The ``value`` input itself, one float per world, such as a quantity or the state of another term."""


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
def _joint_torques_l2(
    world: int, applied_effort: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)
) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        effort = applied_effort[world, joint_ids[k]]
        total += effort * effort
    return total


joint_torques_l2 = Term(
    torch=lambda applied_effort, joint_ids: applied_effort[:, joint_ids].square().sum(-1), warp=_joint_torques_l2
)
"""Sum of the squared efforts the actuators applied, bound to ``robot.applied_effort``."""


@wp.func
def _joint_acc_l2(world: int, joint_acc: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        acceleration = joint_acc[world, joint_ids[k]]
        total += acceleration * acceleration
    return total


joint_acc_l2 = Term(torch=lambda joint_acc, joint_ids: joint_acc[:, joint_ids].square().sum(-1), warp=_joint_acc_l2)
"""Sum of the squared joint accelerations."""


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


def _joint_deviation_l1_torch(
    joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, joint_ids: torch.Tensor
) -> torch.Tensor:
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


def _joint_pos_limits_torch(
    joint_pos: torch.Tensor, soft_joint_pos_limits: torch.Tensor, joint_ids: torch.Tensor
) -> torch.Tensor:
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


def _lin_vel_z_l2_torch(root_link_pose_w: torch.Tensor, root_com_vel_w: torch.Tensor) -> torch.Tensor:
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


def _ang_vel_xy_l2_torch(root_link_pose_w: torch.Tensor, root_com_vel_w: torch.Tensor) -> torch.Tensor:
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


def _flat_orientation_l2_torch(root_link_pose_w: torch.Tensor, gravity_w: torch.Tensor) -> torch.Tensor:
    gravity = projected_gravity_torch(root_link_pose_w, gravity_w)
    return gravity[:, :2].square().sum(-1)


flat_orientation_l2 = Term(torch=_flat_orientation_l2_torch, warp=_flat_orientation_l2)
"""Squared horizontal components of the gravity direction in the root frame: zero when the root is level."""


@wp.func
def _base_height_l2(world: int, root_link_pose_w: wp.array(dtype=wp.transformf), target_height: float) -> float:
    error = wp.transform_get_translation(root_link_pose_w[world])[2] - target_height
    return error * error


base_height_l2 = Term(
    torch=lambda root_link_pose_w, *, target_height: (root_link_pose_w[:, 2] - target_height).square(),
    warp=_base_height_l2,
)
"""Squared error of the root height from ``target_height``, in the world frame: for flat ground at height zero."""


# Velocity tracking.


@wp.func
def _track_lin_vel_xy_exp(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    std: float,
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    velocity = wp.quat_rotate_inv(quat, wp.spatial_top(root_com_vel_w[world]))

    error_x = command[world, 0] - velocity[0]
    error_y = command[world, 1] - velocity[1]
    return wp.exp(-(error_x * error_x + error_y * error_y) / (std * std))


def _track_lin_vel_xy_exp_torch(
    command: torch.Tensor, root_link_pose_w: torch.Tensor, root_com_vel_w: torch.Tensor, *, std: float
) -> torch.Tensor:
    velocity = math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], root_com_vel_w[:, :3])
    error = (command[:, :2] - velocity[:, :2]).square().sum(-1)
    return torch.exp(-error / std**2)


track_lin_vel_xy_exp = Term(torch=_track_lin_vel_xy_exp_torch, warp=_track_lin_vel_xy_exp)
"""Exponential kernel of the squared error of the root's horizontal velocity, in the root frame, from the command."""


@wp.func
def _track_ang_vel_z_exp(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    std: float,
) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    velocity = wp.quat_rotate_inv(quat, wp.spatial_bottom(root_com_vel_w[world]))

    error = command[world, 2] - velocity[2]
    return wp.exp(-(error * error) / (std * std))


def _track_ang_vel_z_exp_torch(
    command: torch.Tensor, root_link_pose_w: torch.Tensor, root_com_vel_w: torch.Tensor, *, std: float
) -> torch.Tensor:
    velocity = math_utils.quat_apply_inverse(root_link_pose_w[:, 3:7], root_com_vel_w[:, 3:6])
    return torch.exp(-(command[:, 2] - velocity[:, 2]).square() / std**2)


track_ang_vel_z_exp = Term(torch=_track_ang_vel_z_exp_torch, warp=_track_ang_vel_z_exp)
"""Exponential kernel of the squared error of the root's yaw rate, in the root frame, from the command."""


@wp.func
def _track_lin_vel_xy_yaw_frame_exp(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    std: float,
) -> float:
    quat = yaw_quat(wp.transform_get_rotation(root_link_pose_w[world]))
    velocity = wp.quat_rotate_inv(quat, wp.spatial_top(root_com_vel_w[world]))
    error_x = command[world, 0] - velocity[0]
    error_y = command[world, 1] - velocity[1]
    return wp.exp(-(error_x * error_x + error_y * error_y) / (std * std))


def _track_lin_vel_xy_yaw_frame_exp_torch(
    command: torch.Tensor, root_link_pose_w: torch.Tensor, root_com_vel_w: torch.Tensor, *, std: float
) -> torch.Tensor:
    quat = math_utils.yaw_quat(root_link_pose_w[:, 3:])
    velocity = math_utils.quat_apply_inverse(quat, root_com_vel_w[:, :3])
    return torch.exp(-(command[:, :2] - velocity[:, :2]).square().sum(dim=1) / std**2)


track_lin_vel_xy_yaw_frame_exp = Term(torch=_track_lin_vel_xy_yaw_frame_exp_torch, warp=_track_lin_vel_xy_yaw_frame_exp)
"""Exponential tracking of planar linear velocity in the gravity-aligned root yaw frame."""


@wp.func
def _track_ang_vel_z_world_exp(
    world: int, command: wp.array2d(dtype=float), root_com_vel_w: wp.array(dtype=wp.spatial_vectorf), std: float
) -> float:
    error = command[world, 2] - wp.spatial_bottom(root_com_vel_w[world])[2]
    return wp.exp(-(error * error) / (std * std))


track_ang_vel_z_world_exp = Term(
    torch=lambda command, root_com_vel_w, *, std: torch.exp(-(command[:, 2] - root_com_vel_w[:, 5]).square() / std**2),
    warp=_track_ang_vel_z_world_exp,
)
"""Exponential tracking of angular velocity around the world vertical axis."""


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


def contact_mask_torch(
    net_normal_forces_w_history: torch.Tensor, body_ids: torch.Tensor, threshold: float
) -> torch.Tensor:
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


def _undesired_contact_mask_torch(
    net_normal_forces_w_history: torch.Tensor, body_ids: torch.Tensor, *, threshold: float
) -> torch.Tensor:
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


def _feet_air_time_torch(
    current_contact_time: torch.Tensor,
    last_air_time: torch.Tensor,
    command: torch.Tensor,
    body_ids: torch.Tensor,
    *,
    threshold: float,
    step_dt: float,
) -> torch.Tensor:
    contact_time = current_contact_time[:, body_ids]
    first_contact = (contact_time > 0.0) & (contact_time < step_dt + FIRST_CONTACT_TOLERANCE)

    reward = ((last_air_time[:, body_ids] - threshold) * first_contact).sum(-1)
    return reward * (command[:, :2].norm(dim=-1) > 0.1)


feet_air_time = Term(torch=_feet_air_time_torch, warp=_feet_air_time)
"""Sum over the ``body_ids`` feet that touched down this step of their air time minus ``threshold`` [s].

Zero for worlds whose planar velocity ``command`` is at most 0.1 m/s. Bind ``step_dt`` [s] to ``mdp2.step_dt``.
"""


@wp.func
def _feet_air_time_positive_biped(
    world: int,
    current_air_time: wp.array2d(dtype=float),
    current_contact_time: wp.array2d(dtype=float),
    command: wp.array2d(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
    overstay_scale: float,
) -> float:
    if wp.length(wp.vec2f(command[world, 0], command[world, 1])) <= 0.1:
        return 0.0

    contacts = int(0)
    min_duration = threshold
    overstay = float(0.0)
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        air_time = current_air_time[world, body]
        contact_time = current_contact_time[world, body]
        in_contact = contact_time > 0.0
        contacts += int(in_contact)
        min_duration = wp.min(min_duration, wp.where(in_contact, contact_time, air_time))
        overstay += wp.max(air_time - threshold, 0.0)
    reward = wp.where(contacts == 1, min_duration, wp.min(threshold, 0.0))
    return reward - wp.max(overstay_scale, 0.0) * overstay


def _feet_air_time_positive_biped_torch(
    current_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    command: torch.Tensor,
    body_ids: torch.Tensor,
    *,
    threshold: float,
    overstay_scale: float,
) -> torch.Tensor:
    air_time = current_air_time[:, body_ids]
    contact_time = current_contact_time[:, body_ids]
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.sum(dim=1) == 1
    reward = torch.where(single_stance, mode_time.amin(dim=1), 0.0).clamp(max=threshold)
    reward -= max(overstay_scale, 0.0) * (air_time - threshold).clamp(min=0.0).sum(dim=1)
    return reward * (command[:, :2].norm(dim=1) > 0.1)


feet_air_time_positive_biped = Term(torch=_feet_air_time_positive_biped_torch, warp=_feet_air_time_positive_biped)
"""Reward the shorter of the swing and stance times [s] in single support, capped at ``threshold``.

``overstay_scale`` penalizes time spent in the air past the threshold. Both contributions require a planar velocity
command above 0.1 m/s.
"""


@wp.func
def _feet_air_time_variance(
    world: int,
    last_air_time: wp.array2d(dtype=float),
    last_contact_time: wp.array2d(dtype=float),
    command: wp.array2d(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    max_time: float,
) -> float:
    if wp.length(wp.vec2f(command[world, 0], command[world, 1])) <= 0.1:
        return 0.0

    count = body_ids.shape[0]
    mean = wp.vec2f()
    for k in range(count):
        body = body_ids[k]
        mean += wp.vec2f(wp.min(last_air_time[world, body], max_time), wp.min(last_contact_time[world, body], max_time))
    mean /= float(count)

    total = float(0.0)
    for k in range(count):
        body = body_ids[k]
        times = wp.vec2f(wp.min(last_air_time[world, body], max_time), wp.min(last_contact_time[world, body], max_time))
        total += wp.length_sq(times - mean)
    return total / float(count - 1)


def _feet_air_time_variance_torch(
    last_air_time: torch.Tensor,
    last_contact_time: torch.Tensor,
    command: torch.Tensor,
    body_ids: torch.Tensor,
    *,
    max_time: float,
) -> torch.Tensor:
    air_variance = last_air_time[:, body_ids].clamp(max=max_time).var(dim=1)
    contact_variance = last_contact_time[:, body_ids].clamp(max=max_time).var(dim=1)
    return (air_variance + contact_variance) * (command[:, :2].norm(dim=1) > 0.1)


def _validate_feet_air_time_variance(ctx: TermContext, **_params) -> None:
    if ctx.inputs["body_ids"].numel() < 2:
        raise ValueError("'feet_air_time_variance' needs at least two bodies to take a variance across feet.")


feet_air_time_variance = Term(
    torch=_feet_air_time_variance_torch, warp=_feet_air_time_variance, init=_validate_feet_air_time_variance
)
"""Sum of the sample variances [s²] across feet of the completed swing and stance times, with each time capped at
``max_time`` [s] before the variances are taken.

``body_ids`` must select at least two feet. Zero for planar velocity commands at most 0.1 m/s.
"""


@wp.func
def _feet_slide(
    world: int,
    body_com_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    net_normal_forces_w_history: wp.array3d(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    aligned_sensor_ids: wp.array(dtype=float),
) -> float:
    speed = float(0.0)
    for k in range(body_ids.shape[0]):
        sensor_id = int(aligned_sensor_ids[k])
        touching = bool(False)
        for i in range(net_normal_forces_w_history.shape[1]):
            touching = touching or wp.length(net_normal_forces_w_history[world, i, sensor_id]) > 1.0
        if touching:
            velocity = wp.spatial_top(body_com_vel_w[world, body_ids[k]])
            speed += wp.length(wp.vec2f(velocity[0], velocity[1]))
    return speed


def _feet_slide_torch(
    body_com_vel_w: torch.Tensor,
    net_normal_forces_w_history: torch.Tensor,
    body_ids: torch.Tensor,
    aligned_sensor_ids: torch.Tensor,
) -> torch.Tensor:
    touching = contact_mask_torch(net_normal_forces_w_history, aligned_sensor_ids.long(), 1.0)
    return (body_com_vel_w[:, body_ids, :2].norm(dim=-1) * touching).sum(dim=1)


def _align_foot_contacts(ctx: TermContext) -> None:
    schema = ctx.scene_data.schema
    body_names = schema.entities[ctx.selectors["body_ids"].entity].names["bodies"]
    sensor_names = schema.entities[ctx.selectors["sensor_ids"].entity].names["bodies"]
    names = [body_names[index] for index in ctx.inputs["body_ids"].tolist()]
    sensor_by_name = {sensor_names[index]: index for index in ctx.inputs["sensor_ids"].tolist()}
    if set(names) != sensor_by_name.keys():
        raise ValueError("'feet_slide' needs matching articulation and contact sensor body names.")
    aligned = ctx.state["aligned_sensor_ids"]
    values = aligned.new_tensor([sensor_by_name[name] for name in names])
    # World configs that define the use differently share this state, which starts as NaN, so they must agree.
    if not aligned.isnan().all() and not torch.equal(aligned, values):
        raise ValueError("'feet_slide': the world configs of a use must select the same feet in the same order.")
    aligned.copy_(values)


feet_slide = Term(
    torch=_feet_slide_torch,
    warp=_feet_slide,
    init=_align_foot_contacts,
    state={"aligned_sensor_ids": State(("body_ids",), initial=float("nan"), persistent=True)},
)
"""Sum the planar speeds of matched articulation feet whose normal force history exceeded 1 N."""


# Pose tracking.


@wp.func
def _position_command_error(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    target_b = wp.vec3f(command[world, 0], command[world, 1], command[world, 2])
    target_w = wp.transform_point(root_link_pose_w[world], target_b)
    position = wp.transform_get_translation(body_link_pose_w[world, body_ids[0]])
    return wp.length(position - target_w)


def _position_command_error_torch(
    command: torch.Tensor, root_link_pose_w: torch.Tensor, body_link_pose_w: torch.Tensor, body_ids: torch.Tensor
) -> torch.Tensor:
    target = root_link_pose_w[:, :3] + math_utils.quat_apply(root_link_pose_w[:, 3:], command[:, :3])
    body = body_link_pose_w[:, body_ids][:, 0, :3]
    return (body - target).norm(dim=1)


position_command_error = Term(torch=_position_command_error_torch, warp=_position_command_error)
"""Position error [m] for the selected body and a pose command in the root frame."""


@wp.func
def _orientation_command_error(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    target_b = wp.quatf(command[world, 3], command[world, 4], command[world, 5], command[world, 6])
    target_w = wp.transform_get_rotation(root_link_pose_w[world]) * target_b
    orientation = wp.transform_get_rotation(body_link_pose_w[world, body_ids[0]])
    return quat_error_magnitude(orientation, target_w)


def _orientation_command_error_torch(
    command: torch.Tensor, root_link_pose_w: torch.Tensor, body_link_pose_w: torch.Tensor, body_ids: torch.Tensor
) -> torch.Tensor:
    target = math_utils.quat_mul(root_link_pose_w[:, 3:], command[:, 3:])
    body = body_link_pose_w[:, body_ids][:, 0, 3:]
    return math_utils.quat_error_magnitude(body, target)


orientation_command_error = Term(torch=_orientation_command_error_torch, warp=_orientation_command_error)
"""Shortest orientation error [rad] for the selected body and a pose command in the root frame."""
