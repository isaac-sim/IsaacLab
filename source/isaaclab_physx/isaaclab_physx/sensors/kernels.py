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


@wp.kernel
def copy_view_rows_to_mesh_table_kernel(
    transforms: wp.array(dtype=wp.transformf),
    view_rows: wp.array2d(dtype=wp.int32),
    mesh_offset: int,
    mesh_positions: wp.array2d(dtype=wp.vec3f),
    mesh_orientations: wp.array2d(dtype=wp.quatf),
):
    """Copy rigid-body view transforms into the per-env mesh table of a ray caster.

    Args:
        transforms: Transforms of the view rows. Shape is (R,).
        view_rows: View row of each table entry of the target, or -1 for entries that keep their poses.
            Shape is (N, M).
        mesh_offset: Table index of the first entry of the target.
        mesh_positions: Mesh positions of the table. Shape is (N, T).
        mesh_orientations: Mesh orientations of the table. Shape is (N, T).
    """
    env, mesh = wp.tid()
    row = view_rows[env, mesh]
    if row < 0:
        return
    mesh_positions[env, mesh_offset + mesh] = wp.transform_get_translation(transforms[row])
    mesh_orientations[env, mesh_offset + mesh] = wp.transform_get_rotation(transforms[row])
