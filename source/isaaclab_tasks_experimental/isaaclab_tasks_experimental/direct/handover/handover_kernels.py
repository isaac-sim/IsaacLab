# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the handover task family.

This is the family's math root: the experimental Direct environment launches
these kernels on the environment's Warp-native state (ProxyArray ``.warp``
views) and its own pre-allocated output buffers. Shared hand idioms (fingertip
segment readers and the quaternion error) come from the reorientation family's
math root; the
reverse dependency never exists.
"""

from __future__ import annotations

import warp as wp

from isaaclab_tasks_experimental.direct.reorient.reorient_kernels import (
    fingertip_pos_col,
    fingertip_quat_col,
    fingertip_vel_col,
)


@wp.kernel
def fall_kernel(
    object_pos_w: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    fall_height: float,
    fell: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    fell[i] = (object_pos_w[i][2] - env_origins[i][2]) <= fall_height


@wp.kernel
def hand_proprio_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
    vel_scale: float,
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    finger_ids: wp.array(dtype=wp.int32),
    env_origins: wp.array(dtype=wp.vec3f),
    actions: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    """Per-hand proprioceptive block, matching the torch concatenation order.

    Launched over ``(num_envs, obs_dim)``: each thread walks a branch ladder over the
    segment boundaries and writes one output column, so warps (32 consecutive columns)
    stay branch-uniform except at segment boundaries.
    """
    i, j = wp.tid()
    num_joints = joint_pos.shape[1]
    num_fingers = finger_ids.shape[0]
    # segment boundaries, in column order
    end_joint = 2 * num_joints
    end_tip_pos = end_joint + 3 * num_fingers
    end_tip_quat = end_tip_pos + 4 * num_fingers
    end_tip_vel = end_tip_quat + 6 * num_fingers
    # hand: normalized DOF positions, scaled DOF velocities
    if j < num_joints:
        out[i, j] = 2.0 * (joint_pos[i, j] - lower[i, j]) / (upper[i, j] - lower[i, j]) - 1.0
    elif j < end_joint:
        out[i, j] = vel_scale * joint_vel[i, j - num_joints]
    # fingertips: environment-frame positions, rotations, spatial velocities
    elif j < end_tip_pos:
        out[i, j] = fingertip_pos_col(body_pos_w, env_origins, finger_ids, i, j - end_joint)
    elif j < end_tip_quat:
        out[i, j] = fingertip_quat_col(body_quat_w, finger_ids, i, j - end_tip_pos)
    elif j < end_tip_vel:
        out[i, j] = fingertip_vel_col(body_vel_w, finger_ids, i, j - end_tip_quat)
    # actions
    else:
        out[i, j] = actions[i, j - end_tip_vel]


@wp.kernel
def handover_reward_kernel(
    goal_distance: wp.array(dtype=wp.float32),
    distance_scale: float,
    reward: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # Direct handover reward: exponential falloff with object-goal distance
    reward[i] = 2.0 * wp.exp(-distance_scale * goal_distance[i])


@wp.kernel
def handover_success_kernel(
    object_pos_w: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    goal_pos_e: wp.array(dtype=wp.vec3f),
    success_distance_threshold: float,
    success: wp.array(dtype=wp.bool),
    goal_distance: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # object position is world-frame; the goal is authored in the environment frame
    distance = wp.length(object_pos_w[i] - env_origins[i] - goal_pos_e[i])
    goal_distance[i] = distance
    success[i] = distance < success_distance_threshold


@wp.kernel
def object_goal_kernel(
    object_pos: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    object_quat: wp.array(dtype=wp.quatf),
    lin_vel: wp.array(dtype=wp.vec3f),
    ang_vel: wp.array(dtype=wp.vec3f),
    goal_pos: wp.array(dtype=wp.vec3f),
    goal_quat: wp.array(dtype=wp.quatf),
    ang_vel_scale: float,
    out: wp.array2d(dtype=wp.float32),
):
    """Assemble one environment's 24-dimensional object/goal observation row."""
    i = wp.tid()
    pos = object_pos[i] - env_origins[i]
    q1 = object_quat[i]
    q2 = goal_quat[i]
    # quat_inverse == conjugate for these unit quaternions, matching
    # isaaclab.utils.math.quat_mul/quat_conjugate semantics
    qe = q1 * wp.quat_inverse(q2)
    out[i, 0] = pos[0]
    out[i, 1] = pos[1]
    out[i, 2] = pos[2]
    out[i, 3] = q1[0]
    out[i, 4] = q1[1]
    out[i, 5] = q1[2]
    out[i, 6] = q1[3]
    out[i, 7] = lin_vel[i][0]
    out[i, 8] = lin_vel[i][1]
    out[i, 9] = lin_vel[i][2]
    out[i, 10] = ang_vel_scale * ang_vel[i][0]
    out[i, 11] = ang_vel_scale * ang_vel[i][1]
    out[i, 12] = ang_vel_scale * ang_vel[i][2]
    out[i, 13] = goal_pos[i][0]
    out[i, 14] = goal_pos[i][1]
    out[i, 15] = goal_pos[i][2]
    out[i, 16] = q2[0]
    out[i, 17] = q2[1]
    out[i, 18] = q2[2]
    out[i, 19] = q2[3]
    out[i, 20] = qe[0]
    out[i, 21] = qe[1]
    out[i, 22] = qe[2]
    out[i, 23] = qe[3]
