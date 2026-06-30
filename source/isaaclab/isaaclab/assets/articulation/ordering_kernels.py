# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import warp as wp


@wp.kernel
def reorder_2d_backend_to_user(
    backend_data: wp.array2d(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    user_data: wp.array2d(dtype=Any),
) -> None:
    """Copy a 2-D backend-order buffer into a user-order buffer."""
    env_id, user_id = wp.tid()
    backend_id = user_to_backend[user_id]
    user_data[env_id, user_id] = backend_data[env_id, backend_id]


@wp.kernel
def write_scalar_user_to_backend_with_indices(
    value: wp.float32,
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Write an indexed scalar into user and backend-order joint buffers."""
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]
    user_data[env_id, user_id] = value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = value


@wp.kernel
def write_scalar_user_to_backend_with_mask(
    value: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Write a masked scalar into user and backend-order joint buffers."""
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    user_data[env_id, user_id] = value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = value


@wp.kernel
def reorder_2d_user_to_backend(
    user_data: wp.array2d(dtype=Any),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_data: wp.array2d(dtype=Any),
) -> None:
    """Copy a 2-D user-order buffer into a backend-order buffer."""
    env_id, backend_id = wp.tid()
    user_id = backend_to_user[backend_id]
    backend_data[env_id, backend_id] = user_data[env_id, user_id]


@wp.kernel
def reorder_body_wrench_user_to_backend(
    user_force: wp.array2d(dtype=wp.vec3f),
    user_torque: wp.array2d(dtype=wp.vec3f),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_force: wp.array2d(dtype=wp.vec3f),
    backend_torque: wp.array2d(dtype=wp.vec3f),
) -> None:
    """Reorder public body-frame force and torque into backend body order."""
    env_id, backend_body_id = wp.tid()
    user_body_id = backend_to_user[backend_body_id]
    backend_force[env_id, backend_body_id] = user_force[env_id, user_body_id]
    backend_torque[env_id, backend_body_id] = user_torque[env_id, user_body_id]


@wp.kernel
def reorder_3d_user_to_backend(
    user_data: wp.array3d(dtype=Any),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_data: wp.array3d(dtype=Any),
) -> None:
    """Copy a 3-D user-order buffer into a backend-order buffer."""
    env_id, backend_id, component_id = wp.tid()
    user_id = backend_to_user[backend_id]
    backend_data[env_id, backend_id, component_id] = user_data[env_id, user_id, component_id]


@wp.kernel
def reorder_3d_backend_to_user(
    backend_data: wp.array3d(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    user_data: wp.array3d(dtype=Any),
) -> None:
    """Copy a 3-D backend-order buffer into a user-order buffer."""
    env_id, user_id, component_id = wp.tid()
    backend_id = user_to_backend[user_id]
    user_data[env_id, user_id, component_id] = backend_data[env_id, backend_id, component_id]


@wp.kernel
def reorder_jacobian_backend_to_user(
    backend_data: wp.array4d(dtype=wp.float32),
    jacobian_body_user_to_backend: wp.array(dtype=wp.int32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_body_ordering: bool,
    has_joint_ordering: bool,
    user_data: wp.array4d(dtype=wp.float32),
) -> None:
    """Copy a backend-order Jacobian into user body and joint order."""
    env_id, user_body_id, spatial_id, user_dof_id = wp.tid()
    backend_body_id = user_body_id
    if has_body_ordering:
        backend_body_id = jacobian_body_user_to_backend[user_body_id]

    backend_dof_id = user_dof_id
    if has_joint_ordering and user_dof_id >= num_base_dofs:
        backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]

    user_data[env_id, user_body_id, spatial_id, user_dof_id] = backend_data[
        env_id, backend_body_id, spatial_id, backend_dof_id
    ]


@wp.kernel
def reorder_mass_matrix_backend_to_user(
    backend_data: wp.array3d(dtype=wp.float32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    user_data: wp.array3d(dtype=wp.float32),
) -> None:
    """Copy a backend-order generalized mass matrix into user joint order."""
    env_id, user_row_id, user_col_id = wp.tid()
    backend_row_id = user_row_id
    backend_col_id = user_col_id
    if has_joint_ordering:
        if user_row_id >= num_base_dofs:
            backend_row_id = num_base_dofs + joint_user_to_backend[user_row_id - num_base_dofs]
        if user_col_id >= num_base_dofs:
            backend_col_id = num_base_dofs + joint_user_to_backend[user_col_id - num_base_dofs]

    user_data[env_id, user_row_id, user_col_id] = backend_data[env_id, backend_row_id, backend_col_id]


@wp.kernel
def reorder_generalized_vector_backend_to_user(
    backend_data: wp.array2d(dtype=wp.float32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Copy a backend-order generalized vector into user joint order."""
    env_id, user_dof_id = wp.tid()
    backend_dof_id = user_dof_id
    if has_joint_ordering and user_dof_id >= num_base_dofs:
        backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]

    user_data[env_id, user_dof_id] = backend_data[env_id, backend_dof_id]


@wp.kernel
def write_joint_vel_user_to_backend_with_indices(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write indexed user-order joint velocities into user and backend-order buffers."""
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    if full_data:
        value = in_data[env_id, user_id]
    else:
        value = in_data[i, j]

    user_vel[env_id, user_id] = value
    user_prev_vel[env_id, user_id] = value
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_vel[env_id, backend_id] = value


@wp.kernel
def write_joint_state_user_to_backend_with_indices(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_pos: wp.array2d(dtype=wp.float32),
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_pos: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write indexed user-order joint state into user and backend-order buffers."""
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    if full_data:
        position = pos_data[env_id, user_id]
        velocity = vel_data[env_id, user_id]
    else:
        position = pos_data[i, j]
        velocity = vel_data[i, j]

    user_pos[env_id, user_id] = position
    user_vel[env_id, user_id] = velocity
    user_prev_vel[env_id, user_id] = velocity
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_pos[env_id, backend_id] = position
        backend_vel[env_id, backend_id] = velocity


@wp.kernel
def write_joint_vel_user_to_backend_with_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write masked user-order joint velocities into user and backend-order buffers."""
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    value = in_data[env_id, user_id]
    user_vel[env_id, user_id] = value
    user_prev_vel[env_id, user_id] = value
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_vel[env_id, backend_id] = value


@wp.kernel
def write_joint_state_user_to_backend_with_mask(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_pos: wp.array2d(dtype=wp.float32),
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_pos: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write masked user-order joint state into user and backend-order buffers."""
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    position = pos_data[env_id, user_id]
    velocity = vel_data[env_id, user_id]
    user_pos[env_id, user_id] = position
    user_vel[env_id, user_id] = velocity
    user_prev_vel[env_id, user_id] = velocity
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_pos[env_id, backend_id] = position
        backend_vel[env_id, backend_id] = velocity


@wp.kernel
def write_2d_float_user_to_backend_with_indices(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Write indexed user-order float data into user and backend-order buffers."""
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    if full_data:
        value = in_data[env_id, user_id]
    else:
        value = in_data[i, j]

    user_data[env_id, user_id] = value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = value


@wp.kernel
def write_2d_float_user_to_backend_with_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Write masked user-order float data into user and backend-order buffers."""
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    value = in_data[env_id, user_id]
    user_data[env_id, user_id] = value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = value
