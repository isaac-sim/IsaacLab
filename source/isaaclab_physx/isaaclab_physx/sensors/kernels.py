# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared warp kernels for PhysX sensors."""

import warp as wp


@wp.kernel
def concat_pos_and_quat_to_pose_kernel(
    pos: wp.array2d(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    pose: wp.array2d(dtype=wp.transformf),
):
    """Concatenate 2D position and quaternion arrays to pose.

    Args:
        pos: Position array. Shape is (N, B).
        quat: Quaternion array. Shape is (N, B).
        pose: Pose array. Shape is (N, B).
    """
    env, sensor = wp.tid()
    pose[env, sensor] = wp.transform(pos[env, sensor], quat[env, sensor])


@wp.kernel
def concat_pos_and_quat_to_pose_1d_kernel(
    pos: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    pose: wp.array(dtype=wp.transformf),
):
    """Concatenate 1D position and quaternion arrays to pose.

    Args:
        pos: Position array. Shape is (N,).
        quat: Quaternion array. Shape is (N,).
        pose: Pose array. Shape is (N,).
    """
    env = wp.tid()
    pose[env] = wp.transform(pos[env], quat[env])
