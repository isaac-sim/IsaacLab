# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the PhysX contact sensor."""

import warp as wp

# ---- Copy kernels (flat PhysX view -> structured data buffers) ----


@wp.kernel
def copy_flat_vec3f_to_2d(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
):
    """Copy flat (N*B,) vec3f from PhysX view into (N, B) vec3f data buffer.

    Args:
        src: Flat source array from PhysX view. Shape is (N*B,).
        dst: Destination data buffer. Shape is (N, B).
        env_ids: Environment indices to update. Shape is (num_env_ids,).
        num_bodies: Number of bodies per environment.
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    src_idx = env_id * num_bodies + j
    dst[env_id, j] = src[src_idx]


@wp.kernel
def copy_flat_vec3f_to_3d(
    src: wp.array2d(dtype=wp.vec3f),
    dst: wp.array3d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
):
    """Copy flat (N*B, M) vec3f from PhysX view into (N, B, M) vec3f data buffer.

    Args:
        src: Flat source array from PhysX view. Shape is (N*B, M).
        dst: Destination data buffer. Shape is (N, B, M).
        env_ids: Environment indices to update. Shape is (num_env_ids,).
        num_bodies: Number of bodies per environment.
    """
    i, j, k = wp.tid()
    env_id = env_ids[i]
    src_row = env_id * num_bodies + j
    dst[env_id, j, k] = src[src_row, k]


@wp.kernel
def split_flat_pose_to_pos_quat(
    src: wp.array(dtype=wp.transformf),
    dst_pos: wp.array2d(dtype=wp.vec3f),
    dst_quat: wp.array2d(dtype=wp.quatf),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
):
    """Split flat (N*B,) transformf into (N, B) vec3f pos and (N, B) quatf quat.

    Args:
        src: Flat source array of transforms from PhysX view. Shape is (N*B,).
        dst_pos: Destination position buffer. Shape is (N, B).
        dst_quat: Destination quaternion buffer. Shape is (N, B).
        env_ids: Environment indices to update. Shape is (num_env_ids,).
        num_bodies: Number of bodies per environment.
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    src_idx = env_id * num_bodies + j
    dst_pos[env_id, j] = wp.transform_get_translation(src[src_idx])
    dst_quat[env_id, j] = wp.transform_get_rotation(src[src_idx])


# ---- History kernels (roll + update) ----


@wp.kernel
def roll_and_update_vec3f_3d(
    history: wp.array3d(dtype=wp.vec3f),
    current: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    history_length: wp.int32,
):
    """Roll (N, T, B) vec3f history buffer and insert current (N, B) vec3f at T=0.

    Args:
        history: History buffer. Shape is (N, T, B).
        current: Current data buffer. Shape is (N, B).
        env_ids: Environment indices to update. Shape is (num_env_ids,).
        history_length: Number of history timesteps T.
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    # Roll: shift all entries forward by one (T-1 -> T-2 -> ... -> 1 -> 0)
    for t in range(history_length - 1, 0, -1):
        history[env_id, t, j] = history[env_id, t - 1, j]
    # Update T=0 with current
    history[env_id, 0, j] = current[env_id, j]


@wp.kernel
def roll_and_update_vec3f_4d(
    history: wp.array4d(dtype=wp.vec3f),
    current: wp.array3d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    history_length: wp.int32,
):
    """Roll (N, T, B, M) vec3f history buffer and insert current (N, B, M) vec3f at T=0.

    Args:
        history: History buffer. Shape is (N, T, B, M).
        current: Current data buffer. Shape is (N, B, M).
        env_ids: Environment indices to update. Shape is (num_env_ids,).
        history_length: Number of history timesteps T.
    """
    i, j, k = wp.tid()
    env_id = env_ids[i]
    # Roll: shift all entries forward by one
    for t in range(history_length - 1, 0, -1):
        history[env_id, t, j, k] = history[env_id, t - 1, j, k]
    # Update T=0 with current
    history[env_id, 0, j, k] = current[env_id, j, k]


# ---- Air/contact time kernel ----


@wp.kernel
def compute_air_contact_time(
    net_forces: wp.array2d(dtype=wp.vec3f),
    current_air_time: wp.array2d(dtype=wp.float32),
    current_contact_time: wp.array2d(dtype=wp.float32),
    last_air_time: wp.array2d(dtype=wp.float32),
    last_contact_time: wp.array2d(dtype=wp.float32),
    elapsed_time: wp.array(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    force_threshold: wp.float32,
):
    """Compute air/contact time from net forces.

    Updates all 4 time buffers (current_air_time, current_contact_time,
    last_air_time, last_contact_time) based on the contact state.

    Args:
        net_forces: Net contact forces. Shape is (N, B).
        current_air_time: Current air time buffer. Shape is (N, B).
        current_contact_time: Current contact time buffer. Shape is (N, B).
        last_air_time: Last air time buffer. Shape is (N, B).
        last_contact_time: Last contact time buffer. Shape is (N, B).
        elapsed_time: Time elapsed since last update per env. Shape is (num_env_ids,).
        env_ids: Environment indices. Shape is (num_env_ids,).
        force_threshold: Force threshold for contact detection.
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    dt = elapsed_time[i]

    is_contact = wp.length(net_forces[env_id, j]) > force_threshold
    cur_air = current_air_time[env_id, j]
    cur_contact = current_contact_time[env_id, j]

    is_first_contact = (cur_air > 0.0) and is_contact
    is_first_detached = (cur_contact > 0.0) and (not is_contact)

    # Update last air time if body has just come into contact
    if is_first_contact:
        last_air_time[env_id, j] = cur_air + dt
    # Update last contact time if body has just detached
    if is_first_detached:
        last_contact_time[env_id, j] = cur_contact + dt

    # Increment time for bodies not in contact, zero if in contact
    if not is_contact:
        current_air_time[env_id, j] = cur_air + dt
    else:
        current_air_time[env_id, j] = 0.0

    # Increment time for bodies in contact, zero if not in contact
    if is_contact:
        current_contact_time[env_id, j] = cur_contact + dt
    else:
        current_contact_time[env_id, j] = 0.0


# ---- Unpack contact buffer data kernel ----


@wp.kernel
def unpack_contact_buffer_data(
    contact_data: wp.array(dtype=wp.vec3f),
    buffer_count: wp.array2d(dtype=wp.uint32),
    buffer_start_indices: wp.array2d(dtype=wp.uint32),
    dst: wp.array3d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
    avg: bool,
    default_val: wp.float32,
):
    """Unpack and aggregate contact buffer data for each (env, body, filter) group.

    Each thread handles one (body, filter) pair for one environment. It reads
    `count` contact entries starting at `start_index` and either averages or
    sums them.

    Args:
        contact_data: Flat buffer of contact data. Shape is (total_contacts,) vec3f.
        buffer_count: Count of contacts per (env*body, filter). Shape is (N*B, M) uint32.
        buffer_start_indices: Start indices per (env*body, filter). Shape is (N*B, M) uint32.
        dst: Destination buffer. Shape is (N, B, M).
        env_ids: Environment indices. Shape is (num_env_ids,).
        num_bodies: Number of bodies per environment.
        avg: If True, average the data; if False, sum it.
        default_val: Default value for groups with zero contacts (e.g. NaN or 0.0).
    """
    i, j, k = wp.tid()
    env_id = env_ids[i]
    flat_idx = env_id * num_bodies + j
    count = wp.int32(buffer_count[flat_idx, k])
    start = wp.int32(buffer_start_indices[flat_idx, k])

    if count > 0:
        accum = wp.vec3f(0.0, 0.0, 0.0)
        for c in range(count):
            accum = accum + contact_data[start + c]
        if avg:
            accum = accum / wp.float32(count)
        dst[env_id, j, k] = accum
    else:
        dst[env_id, j, k] = wp.vec3f(default_val, default_val, default_val)


# ---- Reset kernels ----


@wp.kernel
def reset_vec3f_2d(
    buf: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    val: wp.vec3f,
):
    """Reset (N, B) vec3f buffer for specific env_ids.

    Args:
        buf: Buffer to reset. Shape is (N, B).
        env_ids: Environment indices to reset. Shape is (num_env_ids,).
        val: Value to fill with.
    """
    i, j = wp.tid()
    buf[env_ids[i], j] = val


@wp.kernel
def reset_vec3f_3d(
    buf: wp.array3d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    val: wp.vec3f,
):
    """Reset (N, D1, D2) vec3f buffer for specific env_ids.

    Works for both (N, B, M) and (N, T, B) shaped buffers.

    Args:
        buf: Buffer to reset. Shape is (N, D1, D2).
        env_ids: Environment indices to reset. Shape is (num_env_ids,).
        val: Value to fill with.
    """
    i, j, k = wp.tid()
    buf[env_ids[i], j, k] = val


@wp.kernel
def reset_vec3f_4d(
    buf: wp.array4d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    val: wp.vec3f,
):
    """Reset (N, T, B, M) vec3f buffer for specific env_ids.

    Args:
        buf: Buffer to reset. Shape is (N, T, B, M).
        env_ids: Environment indices to reset. Shape is (num_env_ids,).
        val: Value to fill with.
    """
    i, j, k, m = wp.tid()
    buf[env_ids[i], j, k, m] = val


@wp.kernel
def reset_float_2d(
    buf: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    val: wp.float32,
):
    """Reset (N, B) float32 buffer for specific env_ids.

    Args:
        buf: Buffer to reset. Shape is (N, B).
        env_ids: Environment indices to reset. Shape is (num_env_ids,).
        val: Value to fill with.
    """
    i, j = wp.tid()
    buf[env_ids[i], j] = val


@wp.kernel
def reset_quatf_2d(
    buf: wp.array2d(dtype=wp.quatf),
    env_ids: wp.array(dtype=wp.int32),
    val: wp.quatf,
):
    """Reset (N, B) quatf buffer for specific env_ids.

    Args:
        buf: Buffer to reset. Shape is (N, B).
        env_ids: Environment indices to reset. Shape is (num_env_ids,).
        val: Value to fill with.
    """
    i, j = wp.tid()
    buf[env_ids[i], j] = val
