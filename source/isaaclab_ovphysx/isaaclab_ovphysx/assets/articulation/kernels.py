# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the ovphysx articulation."""

import warp as wp


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
