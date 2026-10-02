# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graph-safe kernels computing task-space controller inputs from simulation-bound articulation buffers."""

from __future__ import annotations

import warp as wp

from isaaclab_newton.assets.kernels import get_link_vel_from_root_com_vel_func


@wp.kernel(enable_backward=False)
def task_space_state(
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_com_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_index: int,
    offset: wp.transformf,
    ee_pose_b: wp.array(dtype=wp.transformf),
    ee_vel_b: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute the target-frame pose and twist relative to the root, in the root frame.

    The twist is the difference of the link and root velocities rotated into the root frame, with the offset's lever
    arm added to the linear part.
    """
    i = wp.tid()
    root = root_link_pose_w[i]
    root_rot_inv = wp.quat_inverse(wp.transform_get_rotation(root))
    link = body_link_pose_w[i, body_index]
    link_pos_b = wp.quat_rotate(root_rot_inv, wp.transform_get_translation(link) - wp.transform_get_translation(root))
    link_rot_b = root_rot_inv * wp.transform_get_rotation(link)
    offset_pos = wp.transform_get_translation(offset)
    ee_pose_b[i] = wp.transformf(
        link_pos_b + wp.quat_rotate(link_rot_b, offset_pos), link_rot_b * wp.transform_get_rotation(offset)
    )

    root_vel = get_link_vel_from_root_com_vel_func(root_com_vel_w[i], root, body_com_pos_b[i, 0])
    link_vel = get_link_vel_from_root_com_vel_func(body_com_vel_w[i, body_index], link, body_com_pos_b[i, body_index])
    lin_b = wp.quat_rotate(root_rot_inv, wp.spatial_top(link_vel) - wp.spatial_top(root_vel))
    ang_b = wp.quat_rotate(root_rot_inv, wp.spatial_bottom(link_vel) - wp.spatial_bottom(root_vel))
    ee_vel_b[i] = wp.spatial_vector(lin_b + wp.cross(ang_b, wp.quat_rotate(link_rot_b, offset_pos)), ang_b)


@wp.kernel(enable_backward=False)
def task_space_jacobian(
    root_link_pose_w: wp.array(dtype=wp.transformf),
    jacobian_w: wp.array4d(dtype=wp.float32),
    jacobian_body: int,
    columns: wp.array(dtype=wp.int32),
    offset: wp.transformf,
    jacobian_b: wp.array3d(dtype=wp.float32),
):
    """Rotate the selected Jacobian columns of one body into the root frame and move them to the offset frame."""
    i, j = wp.tid()
    col = columns[j]
    root_rot_inv = wp.quat_inverse(wp.transform_get_rotation(root_link_pose_w[i]))
    lin = wp.vec3f(
        jacobian_w[i, jacobian_body, 0, col], jacobian_w[i, jacobian_body, 1, col], jacobian_w[i, jacobian_body, 2, col]
    )
    ang = wp.vec3f(
        jacobian_w[i, jacobian_body, 3, col], jacobian_w[i, jacobian_body, 4, col], jacobian_w[i, jacobian_body, 5, col]
    )
    lin = wp.quat_rotate(root_rot_inv, lin)
    ang = wp.quat_rotate(root_rot_inv, ang)
    # v_ee = v_link + w_link x r_link_ee, then rotate the angular rows into the target frame
    lin = lin + wp.cross(ang, wp.transform_get_translation(offset))
    ang = wp.quat_rotate(wp.transform_get_rotation(offset), ang)
    for k in range(3):
        jacobian_b[i, k, j] = lin[k]
        jacobian_b[i, k + 3, j] = ang[k]


@wp.kernel(enable_backward=False)
def task_space_joint_state(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    num_joints: int,
    joint_pos_out: wp.array(dtype=wp.float32),
    joint_vel_out: wp.array(dtype=wp.float32),
):
    """Gather the controlled joints' positions and velocities into compact per-DOF arrays."""
    i, j = wp.tid()
    joint = joint_ids[j]
    joint_pos_out[i * num_joints + j] = joint_pos[i, joint]
    joint_vel_out[i * num_joints + j] = joint_vel[i, joint]


@wp.kernel(enable_backward=False)
def task_space_dynamics(
    mass_matrix: wp.array3d(dtype=wp.float32),
    gravity: wp.array2d(dtype=wp.float32),
    columns: wp.array(dtype=wp.int32),
    num_joints: int,
    use_mass_matrix: bool,
    use_gravity: bool,
    mass_matrix_out: wp.array3d(dtype=wp.float32),
    gravity_out: wp.array(dtype=wp.float32),
):
    """Gather the controlled joints' mass-matrix block and gravity forces."""
    i, j = wp.tid()
    col = columns[j]
    if use_gravity:
        gravity_out[i * num_joints + j] = gravity[i, col]
    if use_mass_matrix:
        for k in range(num_joints):
            mass_matrix_out[i, j, k] = mass_matrix[i, col, columns[k]]


@wp.kernel(enable_backward=False)
def scatter_joint_efforts(
    efforts: wp.array(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    num_joints: int,
    write_joint_act: bool,
    joint_f: wp.array2d(dtype=wp.float32),
    joint_act: wp.array2d(dtype=wp.float32),
):
    """Write compact per-DOF efforts into the articulation's control buffers."""
    i, j = wp.tid()
    effort = efforts[i * num_joints + j]
    joint = joint_ids[j]
    joint_f[i, joint] = effort
    if write_joint_act:
        joint_act[i, joint] = effort
