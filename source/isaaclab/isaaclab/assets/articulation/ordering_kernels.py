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
    backend_id = user_to_backend[user_id] if has_ordering else user_id
    user_data[env_id, user_id] = value
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

    backend_id = user_to_backend[user_id] if has_ordering else user_id
    user_data[env_id, user_id] = value
    backend_data[env_id, backend_id] = value


@wp.kernel
def write_2d_user_to_backend_with_indices(
    in_data: wp.array2d(dtype=Any),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_data: wp.array2d(dtype=Any),
    backend_data: wp.array2d(dtype=Any),
) -> None:
    """Write indexed user-order data into user and backend-order buffers."""
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]
    backend_id = user_to_backend[user_id] if has_ordering else user_id

    if full_data:
        value = in_data[env_id, user_id]
    else:
        value = in_data[i, j]

    user_data[env_id, user_id] = value
    backend_data[env_id, backend_id] = value


@wp.kernel
def write_2d_user_to_backend_with_mask(
    in_data: wp.array2d(dtype=Any),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=Any),
    backend_data: wp.array2d(dtype=Any),
) -> None:
    """Write masked user-order data into user and backend-order buffers."""
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    backend_id = user_to_backend[user_id] if has_ordering else user_id
    value = in_data[env_id, user_id]
    user_data[env_id, user_id] = value
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
    backend_id = user_to_backend[user_id] if has_ordering else user_id

    if full_data:
        value = in_data[env_id, user_id]
    else:
        value = in_data[i, j]

    user_vel[env_id, user_id] = value
    user_prev_vel[env_id, user_id] = value
    user_acc[env_id, user_id] = 0.0
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
    backend_id = user_to_backend[user_id] if has_ordering else user_id

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

    backend_id = user_to_backend[user_id] if has_ordering else user_id
    value = in_data[env_id, user_id]
    user_vel[env_id, user_id] = value
    user_prev_vel[env_id, user_id] = value
    user_acc[env_id, user_id] = 0.0
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

    backend_id = user_to_backend[user_id] if has_ordering else user_id
    position = pos_data[env_id, user_id]
    velocity = vel_data[env_id, user_id]
    user_pos[env_id, user_id] = position
    user_vel[env_id, user_id] = velocity
    user_prev_vel[env_id, user_id] = velocity
    user_acc[env_id, user_id] = 0.0
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
    backend_id = user_to_backend[user_id] if has_ordering else user_id

    if full_data:
        value = in_data[env_id, user_id]
    else:
        value = in_data[i, j]

    user_data[env_id, user_id] = value
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

    backend_id = user_to_backend[user_id] if has_ordering else user_id
    value = in_data[env_id, user_id]
    user_data[env_id, user_id] = value
    backend_data[env_id, backend_id] = value
