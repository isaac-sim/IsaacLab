# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the Kuka Allegro lift task, the MDP2 counterparts of :mod:`isaaclab_tasks.core.lift.mdp`.

Poses are ``wp.transformf`` with ``(x, y, z, w)`` quaternions. Contact terms read ``normal_force_matrix_w`` of the
fingertip sensor, the normal forces of each fingertip against the object.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab_experimental.mdp2 import Implementation, State, Term

from isaaclab.utils import math as math_utils

from .point_cloud import sample_object_point_cloud

# Contacts.


@wp.func
def _contact_force(world: int, forces: wp.array3d(dtype=wp.vec3f), body: int) -> float:
    """Return the magnitude of a fingertip's normal force against the filter objects."""
    total = wp.vec3f(0.0)
    for f in range(forces.shape[2]):
        total = total + forces[world, body, f]
    return wp.length(total)


def _contact_forces_torch(forces, body_ids):
    return forces[:, body_ids].sum(dim=2).norm(dim=-1)


@wp.func
def _max_contact_force(
    world: int, normal_force_matrix_w: wp.array3d(dtype=wp.vec3f), body_ids: wp.array(dtype=wp.int32)
) -> float:
    force = float(0.0)
    for k in range(body_ids.shape[0]):
        force = wp.max(force, _contact_force(world, normal_force_matrix_w, body_ids[k]))
    return force


def _max_contact_force_torch(normal_force_matrix_w, body_ids):
    return _contact_forces_torch(normal_force_matrix_w, body_ids).amax(-1)


max_contact_force = Term(torch=_max_contact_force_torch, warp=_max_contact_force)
"""Largest normal force [N] of the ``body_ids`` fingertips against the object.

The task keeps it as two quantities, ``thumb_force`` and ``finger_force`` of the other fingers: the object is
grasped when both exceed a threshold.
"""


@wp.func
def _grasped(
    world: int, thumb_force: wp.array(dtype=float), finger_force: wp.array(dtype=float), threshold: float
) -> bool:
    """Return whether the thumb and at least one other finger touch the object harder than ``threshold``."""
    return thumb_force[world] > threshold and finger_force[world] > threshold


def _grasped_torch(thumb_force, finger_force, threshold):
    return (thumb_force > threshold) & (finger_force > threshold)


@wp.func
def _contacts(
    world: int, thumb_force: wp.array(dtype=float), finger_force: wp.array(dtype=float), threshold: float
) -> float:
    return wp.where(_grasped(world, thumb_force, finger_force, threshold), 1.0, 0.0)


def _contacts_torch(thumb_force, finger_force, *, threshold):
    return _grasped_torch(thumb_force, finger_force, threshold).float()


contacts = Term(torch=_contacts_torch, warp=_contacts)
"""1 where the thumb and at least one other finger touch the object harder than ``threshold``, else 0."""


@wp.func
def _contact_count(
    world: int, normal_force_matrix_w: wp.array3d(dtype=wp.vec3f), body_ids: wp.array(dtype=wp.int32), threshold: float
) -> float:
    count = float(0.0)
    for k in range(body_ids.shape[0]):
        if _contact_force(world, normal_force_matrix_w, body_ids[k]) > threshold:
            count += 1.0
    return count / float(body_ids.shape[0])


def _contact_count_torch(normal_force_matrix_w, body_ids, *, threshold):
    return (_contact_forces_torch(normal_force_matrix_w, body_ids) > threshold).float().mean(-1)


contact_count = Term(torch=_contact_count_torch, warp=_contact_count)
"""Fraction of the ``body_ids`` fingertips that touch the object harder than ``threshold``."""


# Object and goal.


@wp.func
def _goal_error(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    object_pose_w: wp.array(dtype=wp.transformf),
) -> float:
    goal = wp.transform_point(
        root_link_pose_w[world], wp.vec3f(command[world, 0], command[world, 1], command[world, 2])
    )
    return wp.length(wp.transform_get_translation(object_pose_w[world]) - goal)


def _goal_error_torch(command, root_link_pose_w, object_pose_w):
    goal, _ = math_utils.combine_frame_transforms(root_link_pose_w[:, :3], root_link_pose_w[:, 3:7], command[:, :3])
    return (object_pose_w[:, :3] - goal).norm(dim=-1)


goal_error = Term(torch=_goal_error_torch, warp=_goal_error)
"""Distance [m] from the object to the commanded position, which is in the robot's root frame."""


@wp.func
def _succeeded(
    world: int,
    goal_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    succeeded: wp.array(dtype=float),
    distance: float,
    contact_threshold: float,
) -> float:
    if goal_error[world] < distance and _grasped(world, thumb_force, finger_force, contact_threshold):
        return 1.0
    return succeeded[world]


@wp.func
def _succeeded_update(
    world: int,
    goal_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    succeeded: wp.array(dtype=float),
    distance: float,
    contact_threshold: float,
):
    succeeded[world] = _succeeded(world, goal_error, thumb_force, finger_force, succeeded, distance, contact_threshold)


def _succeeded_torch(goal_error, thumb_force, finger_force, succeeded, *, distance, contact_threshold):
    reached = (goal_error < distance) & _grasped_torch(thumb_force, finger_force, contact_threshold)
    return torch.where(reached, 1.0, succeeded)


def _succeeded_update_torch(succeeded, mask=None, **arguments):
    succeeded.copy_(_succeeded_torch(succeeded=succeeded, **arguments))


succeeded = Term(
    torch=Implementation(compute=_succeeded_torch, update=_succeeded_update_torch),
    warp=Implementation(compute=_succeeded, update=_succeeded_update),
    state={"succeeded": State()},
)
"""1 once the grasped object has been within ``distance`` of the goal in the episode, else 0.

A quantity that the curriculum and the reset bank read when the episode ends.
"""


@wp.func
def _object_ee_distance(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
    object_pose_w: wp.array(dtype=wp.transformf),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    std: float,
    contact_threshold: float,
) -> float:
    object_position = wp.transform_get_translation(object_pose_w[world])
    distance = float(0.0)
    for k in range(body_ids.shape[0]):
        body_position = wp.transform_get_translation(body_link_pose_w[world, body_ids[k]])
        distance = wp.max(distance, wp.length(body_position - object_position))

    grasped = _grasped(world, thumb_force, finger_force, contact_threshold)
    return (1.0 - wp.tanh(distance / std)) * wp.where(grasped, 1.0, 0.1)


def _object_ee_distance_torch(
    body_link_pose_w, body_ids, object_pose_w, thumb_force, finger_force, *, std, contact_threshold
):
    distance = (body_link_pose_w[:, body_ids, :3] - object_pose_w[:, None, :3]).norm(dim=-1).amax(dim=-1)
    grasped = _grasped_torch(thumb_force, finger_force, contact_threshold)
    return (1.0 - torch.tanh(distance / std)) * torch.where(grasped, 1.0, 0.1)


object_ee_distance = Term(torch=_object_ee_distance_torch, warp=_object_ee_distance)
"""``1 - tanh(d / std)`` of the largest distance ``d`` from the ``body_ids`` bodies to the object.

Scaled by 0.1 unless the object is grasped with forces above ``contact_threshold``.
"""


@wp.func
def _progress(
    world: int,
    command: wp.array2d(dtype=float),
    goal_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    best_error: wp.array(dtype=float),
    last_command: wp.array2d(dtype=float),
    min_improvement: float,
    contact_threshold: float,
) -> wp.vec2f:
    """Return whether the step is paid, and the best error so far, which restarts when the command changes."""
    error = goal_error[world]
    best = best_error[world]
    for k in range(command.shape[1]):
        if command[world, k] != last_command[world, k]:
            best = wp.inf
    if best == wp.inf:
        best = error

    if _grasped(world, thumb_force, finger_force, contact_threshold) and error < best - min_improvement:
        return wp.vec2f(1.0, error)
    return wp.vec2f(0.0, best)


@wp.func
def _position_command_progress(
    world: int,
    command: wp.array2d(dtype=float),
    goal_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    best_error: wp.array(dtype=float),
    last_command: wp.array2d(dtype=float),
    min_improvement: float,
    contact_threshold: float,
) -> float:
    progress = _progress(
        world,
        command,
        goal_error,
        thumb_force,
        finger_force,
        best_error,
        last_command,
        min_improvement,
        contact_threshold,
    )
    return progress[0]


@wp.func
def _position_command_progress_update(
    world: int,
    command: wp.array2d(dtype=float),
    goal_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    best_error: wp.array(dtype=float),
    last_command: wp.array2d(dtype=float),
    min_improvement: float,
    contact_threshold: float,
):
    progress = _progress(
        world,
        command,
        goal_error,
        thumb_force,
        finger_force,
        best_error,
        last_command,
        min_improvement,
        contact_threshold,
    )
    best_error[world] = progress[1]
    for k in range(command.shape[1]):
        last_command[world, k] = command[world, k]


def _progress_torch(
    command, goal_error, thumb_force, finger_force, best_error, last_command, *, min_improvement, contact_threshold
):
    restart = (command != last_command).any(-1) | best_error.isinf()
    best = torch.where(restart, goal_error, best_error)
    paid = _grasped_torch(thumb_force, finger_force, contact_threshold) & (goal_error < best - min_improvement)
    return paid, torch.where(paid, goal_error, best)


def _position_command_progress_torch(**arguments):
    paid, _ = _progress_torch(**arguments)
    return paid.float()


def _position_command_progress_update_torch(mask=None, **arguments):
    _, best = _progress_torch(**arguments)
    arguments["best_error"].copy_(best)
    arguments["last_command"].copy_(arguments["command"])


position_command_progress = Term(
    torch=Implementation(compute=_position_command_progress_torch, update=_position_command_progress_update_torch),
    warp=Implementation(compute=_position_command_progress, update=_position_command_progress_update),
    state={"best_error": State(initial=float("inf")), "last_command": State(shape=("worlds", "command"))},
)
"""1 each time the grasped object gets ``min_improvement`` closer to the commanded position than ever before.

The best error restarts on episode resets and when the command changes, so ground already credited cannot be earned
again by backing off.
"""


@wp.func
def _success_reward(
    world: int,
    goal_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    pos_std: float,
    contact_threshold: float,
) -> float:
    if not _grasped(world, thumb_force, finger_force, contact_threshold):
        return 0.0
    closeness = 1.0 - wp.tanh(goal_error[world] / pos_std)
    return closeness * closeness


def _success_reward_torch(goal_error, thumb_force, finger_force, *, pos_std, contact_threshold):
    closeness = 1.0 - torch.tanh(goal_error / pos_std)
    return closeness.square() * _grasped_torch(thumb_force, finger_force, contact_threshold)


success_reward = Term(torch=_success_reward_torch, warp=_success_reward)
"""``(1 - tanh(e / pos_std))^2`` of the goal position error ``e``, while the object is grasped.

MDP1's orientation factor is left out: the lift task tracks the position only.
"""


@wp.func
def _abnormal_robot(
    world: int,
    joint_vel: wp.array2d(dtype=float),
    soft_joint_vel_limits: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        if wp.abs(joint_vel[world, j]) > soft_joint_vel_limits[world, j]:
            return 1.0
    return 0.0


def _abnormal_robot_torch(joint_vel, soft_joint_vel_limits, joint_ids):
    return (joint_vel[:, joint_ids].abs() > soft_joint_vel_limits[:, joint_ids]).any(-1).float()


abnormal_robot = Term(torch=_abnormal_robot_torch, warp=_abnormal_robot)
"""1 where a joint speed exceeds its soft limit, a quantity that both ends the episode and is penalized.

MDP1 rewards the termination itself, except where the episode also times out; this term ignores time-outs.
"""


@wp.func
def _out_of_bound(
    world: int,
    object_pose_w: wp.array(dtype=wp.transformf),
    object_vel_w: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    progress: wp.array(dtype=float),
    in_bound_range: wp.types.matrix(shape=(3, 2), dtype=wp.float32),
    final_in_bound_range: wp.types.matrix(shape=(3, 2), dtype=wp.float32),
) -> bool:
    bounds = in_bound_range + (final_in_bound_range - in_bound_range) * progress[world]
    pose = object_pose_w[world]
    velocity = object_vel_w[world]
    position = wp.transform_get_translation(pose) - env_origins[world]
    rotation = wp.transform_get_rotation(pose)

    outside = bool(False)
    for axis in range(3):
        outside = outside or position[axis] < bounds[axis, 0] or position[axis] > bounds[axis, 1]
        outside = outside or not wp.isfinite(position[axis])
        outside = outside or not wp.isfinite(velocity[axis]) or not wp.isfinite(velocity[3 + axis])
    for k in range(4):
        outside = outside or not wp.isfinite(rotation[k])
    return outside


def _out_of_bound_torch(object_pose_w, object_vel_w, env_origins, progress, *, in_bound_range, final_in_bound_range):
    position = object_pose_w[:, :3] - env_origins
    initial, final = position.new_tensor(in_bound_range), position.new_tensor(final_in_bound_range)
    bounds = initial + (final - initial) * progress[:, None, None]
    outside = ((position < bounds[..., 0]) | (position > bounds[..., 1])).any(-1)
    finite = object_pose_w.isfinite().all(-1) & object_vel_w.isfinite().all(-1)
    return outside | ~finite


out_of_bound = Term(torch=_out_of_bound_torch, warp=_out_of_bound)
"""Whether the object left its bounds (x, y, z pairs relative to the environment origin), or its state is not finite.

The bounds move from ``in_bound_range`` to ``final_in_bound_range`` with a curriculum's ``progress`` input.
"""


# Observations.


@wp.func
def _object_quat_b(
    world: int,
    element: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    object_pose_w: wp.array(dtype=wp.transformf),
) -> float:
    robot = wp.transform_get_rotation(root_link_pose_w[world])
    quat = wp.quat_inverse(robot) * wp.transform_get_rotation(object_pose_w[world])
    return quat[element]


def _object_quat_b_torch(root_link_pose_w, object_pose_w):
    return math_utils.quat_mul(math_utils.quat_inv(root_link_pose_w[:, 3:7]), object_pose_w[:, 3:7])


object_quat_b = Term(torch=_object_quat_b_torch, warp=_object_quat_b, shape=("worlds", 4))
"""Orientation of the object in the robot's root frame."""


@wp.func
def _body_state_b(
    world: int,
    body: int,
    element: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    pose = wp.transform_inverse(root_link_pose_w[world]) * body_link_pose_w[world, body_ids[body]]
    if element < 3:
        return wp.transform_get_translation(pose)[element]
    return wp.transform_get_rotation(pose)[element - 3]


def _body_state_b_torch(body_link_pose_w, root_link_pose_w, body_ids):
    body = body_link_pose_w[:, body_ids]
    root = root_link_pose_w[:, None].expand_as(body)
    position, quat = math_utils.subtract_frame_transforms(root[..., :3], root[..., 3:7], body[..., :3], body[..., 3:7])
    return torch.cat([position, quat], dim=-1)


body_state_b = Term(torch=_body_state_b_torch, warp=_body_state_b, shape=("worlds", "body_ids", 7))
"""Pose ``(x, y, z, qx, qy, qz, qw)`` of each of the ``body_ids`` bodies in the robot's root frame."""


@wp.func
def _fingers_contact_force_b(
    world: int,
    body: int,
    axis: int,
    normal_force_matrix_w: wp.array3d(dtype=wp.vec3f),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    total = wp.vec3f(0.0)
    for f in range(normal_force_matrix_w.shape[2]):
        total = total + normal_force_matrix_w[world, body_ids[body], f]
    return wp.quat_rotate_inv(wp.transform_get_rotation(root_link_pose_w[world]), total)[axis]


def _fingers_contact_force_b_torch(normal_force_matrix_w, root_link_pose_w, body_ids):
    forces = normal_force_matrix_w[:, body_ids].sum(dim=2)
    root = root_link_pose_w[:, None, 3:7].expand(-1, forces.shape[1], -1)
    return math_utils.quat_apply_inverse(root, forces)


fingers_contact_force_b = Term(
    torch=_fingers_contact_force_b_torch, warp=_fingers_contact_force_b, shape=("worlds", "body_ids", 3)
)
"""Normal force of each of the ``body_ids`` fingertips against the object, in the robot's root frame."""


def _sample_points(ctx, *, num_points):
    ctx.state["points"].copy_(sample_object_point_cloud(ctx.entity("object_pose_w"), num_points))


@wp.func
def _object_point_cloud_b(
    world: int,
    point: int,
    axis: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    object_pose_w: wp.array(dtype=wp.transformf),
    points: wp.array3d(dtype=float),
    num_points: int,
) -> float:
    local = wp.vec3f(points[world, point, 0], points[world, point, 1], points[world, point, 2])
    position_w = wp.transform_point(object_pose_w[world], local)
    return wp.transform_point(wp.transform_inverse(root_link_pose_w[world]), position_w)[axis]


def _object_point_cloud_b_torch(root_link_pose_w, object_pose_w, points, *, num_points):
    object_pose = object_pose_w[:, None].expand(-1, num_points, -1)
    position_w = math_utils.quat_apply(object_pose[..., 3:7], points) + object_pose[..., :3]
    root = root_link_pose_w[:, None].expand(-1, num_points, -1)
    position, _ = math_utils.subtract_frame_transforms(root[..., :3], root[..., 3:7], position_w)
    return position


object_point_cloud_b = Term(
    torch=_object_point_cloud_b_torch,
    warp=_object_point_cloud_b,
    shape=("worlds", "num_points", 3),
    state={"points": State(shape=("worlds", "num_points", 3), persistent=True)},
    init=_sample_points,
)
"""``num_points`` points on the object's surface, in the robot's root frame.

The points are sampled once, in each object's frame, from the collision shapes of the object each world spawned.
"""


# Curriculum.


@wp.func
def _curriculum_gravity(
    world: int,
    gravity_w: wp.array(dtype=wp.vec3f),
    progress: wp.array(dtype=float),
    initial_gravity: wp.vec3f,
    final_gravity: wp.vec3f,
):
    gravity_w[world] = initial_gravity + (final_gravity - initial_gravity) * progress[world]


def _curriculum_gravity_torch(mask, gravity_w, progress, *, initial_gravity, final_gravity):
    initial, final = gravity_w.new_tensor(initial_gravity), gravity_w.new_tensor(final_gravity)
    value = initial + (final - initial) * progress[:, None]
    gravity_w.copy_(torch.where(mask[:, None], value, gravity_w))


curriculum_gravity = Term(torch=_curriculum_gravity_torch, warp=_curriculum_gravity)
"""Set the gravity of the reset worlds between ``initial_gravity`` and ``final_gravity`` by a curriculum's
``progress`` input, like MDP1's ``randomize_physics_scene_gravity`` with its range moved by the curriculum."""


# Candidate start states, which only the prefill of the reset bank runs, on the Warp backend.


@wp.func
def _reset_to_target(
    world: int,
    object_pose_w: wp.array(dtype=wp.transformf),
    object_vel_w: wp.array(dtype=wp.spatial_vectorf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
    position_range: wp.types.matrix(shape=(3, 2), dtype=wp.float32),
    probability: float,
    rng: wp.int32,
):
    if wp.randf(wp.rand_init(rng, 0)) >= probability:
        return

    offset = wp.vec3f()
    for axis in range(3):
        offset[axis] = wp.randf(wp.rand_init(rng, 1 + axis), position_range[axis, 0], position_range[axis, 1])

    # A uniformly random orientation, from three uniform samples.
    u = wp.randf(wp.rand_init(rng, 4))
    a = 2.0 * wp.pi * wp.randf(wp.rand_init(rng, 5))
    b = 2.0 * wp.pi * wp.randf(wp.rand_init(rng, 6))
    orientation = wp.quatf(
        wp.sqrt(1.0 - u) * wp.sin(a), wp.sqrt(1.0 - u) * wp.cos(a), wp.sqrt(u) * wp.sin(b), wp.sqrt(u) * wp.cos(b)
    )

    position = wp.transform_point(body_link_pose_w[world, body_ids[0]], offset)
    object_pose_w[world] = wp.transformf(position, orientation)
    object_vel_w[world] = wp.spatial_vectorf()


reset_to_target = Term(warp=_reset_to_target, random=True)
"""Place the object at rest on the first of the ``body_ids`` bodies, in a share ``probability`` of the worlds.

Its position is the body's plus a uniform offset from ``position_range`` in the body's frame, and its orientation is
uniformly random. The event must ask for a commit, to read the body pose that the joint reset gave. Draws that
penetrate the hand are for the reset bank's criteria to reject.
"""


@wp.func
def _grasp_travel_distance(
    world: int,
    element: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    object_pose_w: wp.array(dtype=wp.transformf),
    goal_center: wp.vec3f,
) -> float:
    object_position = wp.transform_get_translation(object_pose_w[world])
    if element == 1:
        return wp.length(wp.transform_point(root_link_pose_w[world], goal_center) - object_position)

    distance = float(0.0)
    for k in range(body_ids.shape[0]):
        body_position = wp.transform_get_translation(body_link_pose_w[world, body_ids[k]])
        distance = wp.max(distance, wp.length(body_position - object_position))
    return distance


grasp_travel_distance = Term(warp=_grasp_travel_distance, shape=("worlds", 2))
"""How far the hand has to close on the object, and how far the object then has to travel [m].

The largest distance from the ``body_ids`` bodies to the object, and the distance from the object to ``goal_center``,
the middle of the goal region in the robot's root frame. The reset bank keeps the start states most spread over the
two, which keeps the easy and hard corners that rejection sampling alone leaves rare.
"""
