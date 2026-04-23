# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the ovphysx articulation."""

import warp as wp


@wp.kernel
def _body_wrench_to_world(
    force_b: wp.array(dtype=wp.vec3f, ndim=2),
    torque_b: wp.array(dtype=wp.vec3f, ndim=2),
    poses: wp.array(dtype=wp.transformf, ndim=2),
    wrench_out: wp.array(dtype=wp.float32, ndim=3),
):
    """Rotate body-frame force/torque to world frame and pack into [N, L, 9]."""
    i, j = wp.tid()
    q = wp.transform_get_rotation(poses[i, j])
    f_w = wp.quat_rotate(q, force_b[i, j])
    t_w = wp.quat_rotate(q, torque_b[i, j])
    wrench_out[i, j, 0] = f_w[0]
    wrench_out[i, j, 1] = f_w[1]
    wrench_out[i, j, 2] = f_w[2]
    wrench_out[i, j, 3] = t_w[0]
    wrench_out[i, j, 4] = t_w[1]
    wrench_out[i, j, 5] = t_w[2]
    p_w = wp.transform_get_translation(poses[i, j])
    wrench_out[i, j, 6] = p_w[0]
    wrench_out[i, j, 7] = p_w[1]
    wrench_out[i, j, 8] = p_w[2]


@wp.kernel
def _scatter_rows_partial(
    dst: wp.array2d(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    ids: wp.array(dtype=wp.int32),
):
    """dst[ids[i], j] = src[i, j] -- scatter partial [K,C] into full [N,C] on GPU."""
    i, j = wp.tid()
    dst[ids[i], j] = src[i, j]


@wp.func
def compute_soft_joint_pos_limits_func(
    joint_pos_limits: wp.vec2f,
    soft_limit_factor: wp.float32,
):
    """Compute soft joint position limits from hard limits."""
    joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0
    joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]
    return wp.vec2f(
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
    )


@wp.kernel
def update_soft_joint_pos_limits(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
):
    """Update soft joint position limits from hard limits and a scale factor."""
    i, j = wp.tid()
    soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)


"""
Data-layer kernels (used by ArticulationData).
"""


@wp.kernel
def _fd_joint_acc(
    cur_vel: wp.array2d(dtype=wp.float32),
    prev_vel: wp.array2d(dtype=wp.float32),
    inv_dt: float,
    out: wp.array2d(dtype=wp.float32),
):
    """Compute joint acceleration via finite differencing and update previous velocity.

    Args:
        cur_vel: Current joint velocities. Shape is (num_envs, num_joints).
        prev_vel: Previous joint velocities (updated in-place). Shape is (num_envs, num_joints).
        inv_dt: Inverse time step (1/dt) [1/s].
        out: Output joint accelerations. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out[i, j] = (cur_vel[i, j] - prev_vel[i, j]) * inv_dt
    prev_vel[i, j] = cur_vel[i, j]


@wp.kernel
def _copy_first_body(
    body_vel: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Copy the first body's velocity to the root velocity buffer.

    Args:
        body_vel: Body velocities. Shape is (num_envs, num_bodies).
        root_vel: Output root velocities. Shape is (num_envs,).
    """
    i = wp.tid()
    root_vel[i] = body_vel[i, 0]


@wp.kernel
def _compose_root_com_pose(
    link_pose: wp.array(dtype=wp.transformf),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compose root link pose with body-frame CoM offset to get world-frame root CoM pose.

    Args:
        link_pose: Root link poses in world frame. Shape is (num_envs,).
        com_pose_b: Body-frame CoM offsets. Shape is (num_envs, num_bodies).
        com_pose_w: Output world-frame root CoM poses. Shape is (num_envs,).
    """
    i = wp.tid()
    com_pose_w[i] = wp.transform_multiply(link_pose[i], com_pose_b[i, 0])


@wp.kernel
def _compose_body_com_poses(
    link_pose: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf, ndim=2),
):
    """Compose body link poses with body-frame CoM offsets to get world-frame CoM poses.

    Args:
        link_pose: Body link poses in world frame. Shape is (num_envs, num_bodies).
        com_pose_b: Body-frame CoM offsets. Shape is (num_envs, num_bodies).
        com_pose_w: Output world-frame body CoM poses. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    com_pose_w[i, j] = wp.transform_multiply(link_pose[i, j], com_pose_b[i, j])


@wp.kernel
def _projected_gravity(
    gravity_vec_w: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.vec3f),
):
    """Project world-frame gravity direction into the root body frame.

    Args:
        gravity_vec_w: Gravity unit vector per instance in world frame. Shape is (num_envs,).
        root_pose: Root link poses in world frame. Shape is (num_envs,).
        out: Output projected gravity in body frame. Shape is (num_envs,).
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    out[i] = wp.quat_rotate_inv(q, gravity_vec_w[i])


@wp.kernel
def _compute_heading(
    forward_vec_b: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.float32),
):
    """Compute yaw heading angle from the forward direction rotated into the world frame.

    Args:
        forward_vec_b: Forward direction in body frame per instance. Shape is (num_envs,).
        root_pose: Root link poses in world frame. Shape is (num_envs,).
        out: Output heading angles [rad]. Shape is (num_envs,).
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    forward = wp.quat_rotate(q, forward_vec_b[i])
    out[i] = wp.atan2(forward[1], forward[0])


@wp.kernel
def _world_vel_to_body_lin(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    """Rotate world-frame linear velocity into the root body frame.

    Args:
        root_pose: Root link poses in world frame. Shape is (num_envs,).
        vel_w: Spatial velocities in world frame. Shape is (num_envs,).
        out: Output linear velocity in body frame. Shape is (num_envs,).
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    lin = wp.spatial_top(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, lin)


@wp.kernel
def _world_vel_to_body_ang(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    """Rotate world-frame angular velocity into the root body frame.

    Args:
        root_pose: Root link poses in world frame. Shape is (num_envs,).
        vel_w: Spatial velocities in world frame. Shape is (num_envs,).
        out: Output angular velocity in body frame. Shape is (num_envs,).
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    ang = wp.spatial_bottom(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, ang)
