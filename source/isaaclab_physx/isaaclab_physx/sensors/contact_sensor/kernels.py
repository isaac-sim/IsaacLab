# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the PhysX contact sensor."""

import warp as wp

# ---- Copy kernels (flat PhysX view -> structured data buffers) ----


@wp.kernel
def split_flat_pose_to_pos_quat(
    src: wp.array(dtype=wp.transformf),
    mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    dst_pos: wp.array2d(dtype=wp.vec3f),
    dst_quat: wp.array2d(dtype=wp.quatf),
):
    """Split flat (N*B,) transformf into (N, B) vec3f pos and (N, B) quatf quat.

    Args:
        src: Flat source array of transforms from PhysX view. Shape is (N*B,).
        mask: Boolean mask for which environments to update. Shape is (N,).
        num_bodies: Number of bodies per environment.
        dst_pos: Destination position buffer. Shape is (N, B).
        dst_quat: Destination quaternion buffer. Shape is (N, B).
    """
    env, sensor = wp.tid()
    if mask:
        if not mask[env]:
            return

    src_idx = env * num_bodies + sensor
    dst_pos[env, sensor] = wp.transform_get_translation(src[src_idx])
    dst_quat[env, sensor] = wp.transform_get_rotation(src[src_idx])


# ---- Unpack contact buffer data kernel ----


@wp.kernel
def unpack_contact_buffer_data(
    contact_data: wp.array(dtype=wp.vec3f),
    buffer_count: wp.array2d(dtype=wp.uint32),
    buffer_start_indices: wp.array2d(dtype=wp.uint32),
    mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    avg: bool,
    default_val: wp.float32,
    dst: wp.array3d(dtype=wp.vec3f),
):
    """Unpack and aggregate contact buffer data for each (env, body, filter) group.

    Each thread handles one (body, filter) pair for one environment. It reads
    `count` contact entries starting at `start_index` and either averages or
    sums them.

    Args:
        contact_data: Flat buffer of contact data. Shape is (total_contacts,) vec3f.
        buffer_count: Count of contacts per (env*body, filter). Shape is (N*B, M) uint32.
        buffer_start_indices: Start indices per (env*body, filter). Shape is (N*B, M) uint32.
        mask: Boolean mask for which environments to update. Shape is (N,).
        num_bodies: Number of bodies per environment.
        avg: If True, average the data; if False, sum it.
        default_val: Default value for groups with zero contacts (e.g. NaN or 0.0).
        dst: Destination buffer. Shape is (N, B, M).
    """
    env, sensor, contact = wp.tid()
    if mask:
        if not mask[env]:
            return

    flat_idx = env * num_bodies + sensor
    count = wp.int32(buffer_count[flat_idx, contact])
    start = wp.int32(buffer_start_indices[flat_idx, contact])

    if count > 0:
        accum = wp.vec3f(0.0, 0.0, 0.0)
        for c in range(count):
            accum = accum + contact_data[start + c]
        if avg:
            accum = accum / wp.float32(count)
        dst[env, sensor, contact] = accum
    else:
        dst[env, sensor, contact] = wp.vec3f(default_val, default_val, default_val)


@wp.kernel
def reset_contact_sensor_kernel(
    # in
    history_length: int,
    num_filter_objects: int,
    env_mask: wp.array(dtype=wp.bool),
    # in-out
    net_forces_w: wp.array2d(dtype=wp.vec3f),
    net_forces_w_history: wp.array3d(dtype=wp.vec3f),
    force_matrix_w: wp.array3d(dtype=wp.vec3f),
    # outputs
    current_air_time: wp.array2d(dtype=wp.float32),
    last_air_time: wp.array2d(dtype=wp.float32),
    current_contact_time: wp.array2d(dtype=wp.float32),
    last_contact_time: wp.array2d(dtype=wp.float32),
    friction_forces_w: wp.array3d(dtype=wp.vec3f),
    contact_pos_w: wp.array3d(dtype=wp.vec3f),
):
    """Reset the contact sensor data for specified environments.

    Launch with dim=(num_envs, num_sensors).

    Args:
        history_length: Length of history.
        num_filter_objects: Number of filter objects.
        env_mask: Mask array. Shape is (num_envs,).
        net_forces_w: Net forces array. Shape is (num_envs, num_sensors).
        net_forces_w_history: Net forces history array. Shape is (num_envs, history_length, num_sensors).
        force_matrix_w: Force matrix array. Shape is (num_envs, num_sensors, num_filter_objects).
        current_air_time: Current air time array. Shape is (num_envs, num_sensors).
        last_air_time: Last air time array. Shape is (num_envs, num_sensors).
        current_contact_time: Current contact time array. Shape is (num_envs, num_sensors).
        last_contact_time: Last contact time array. Shape is (num_envs, num_sensors).
        friction_forces_w: Friction forces array. Shape is (num_envs, num_sensors, num_filter_objects).
        contact_pos_w: Contact pos array. Shape is (num_envs, num_sensors, num_filter_objects).
    """
    env, sensor = wp.tid()

    if env_mask:
        if not env_mask[env]:
            return

    # Reset net forces
    net_forces_w[env, sensor] = wp.vec3f(0.0)

    # Reset history
    if net_forces_w_history:
        for i in range(history_length):
            net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)

    # Reset force matrix (guard for None case)
    if force_matrix_w:
        for f in range(num_filter_objects):
            force_matrix_w[env, sensor, f] = wp.vec3f(0.0)

    # Reset air/contact time tracking
    if current_air_time:
        current_air_time[env, sensor] = 0.0
        last_air_time[env, sensor] = 0.0
        current_contact_time[env, sensor] = 0.0
        last_contact_time[env, sensor] = 0.0

    if friction_forces_w:
        for f in range(num_filter_objects):
            friction_forces_w[env, sensor, f] = wp.vec3f(0.0)

    if contact_pos_w:
        for f in range(num_filter_objects):
            contact_pos_w[env, sensor, f] = wp.vec3f(0.0)


@wp.kernel
def compute_first_transition_kernel(
    # in
    threshold: wp.float32,
    time: wp.array2d(dtype=wp.float32),
    # out
    result: wp.array2d(dtype=wp.float32),
):
    """Compute boolean mask (as float) for sensors whose time is in (0, threshold).

    Used by both compute_first_contact (with current_contact_time) and
    compute_first_air (with current_air_time).

    Launch with dim=(num_envs, num_sensors).

    Args:
        threshold: Threshold for the time.
        time: Time array. Shape is (num_envs, num_sensors).
        result: Result array. Shape is (num_envs, num_sensors).
    """
    env, sensor = wp.tid()
    t = time[env, sensor]
    if t > 0.0 and t < threshold:
        result[env, sensor] = 1.0
    else:
        result[env, sensor] = 0.0


@wp.kernel
def update_net_forces_kernel(
    # in
    net_forces_flat: wp.array(dtype=wp.vec3f),
    net_forces_matrix_flat: wp.array2d(dtype=wp.vec3f),
    mask: wp.array(dtype=wp.bool),
    num_sensors: int,
    num_filter_shapes: int,
    history_length: int,
    contact_force_threshold: wp.float32,
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
    # out
    net_forces_w: wp.array2d(dtype=wp.vec3f),
    net_forces_w_history: wp.array3d(dtype=wp.vec3f),
    force_matrix_w: wp.array3d(dtype=wp.vec3f),
    force_matrix_w_history: wp.array4d(dtype=wp.vec3f),
    current_air_time: wp.array2d(dtype=wp.float32),
    current_contact_time: wp.array2d(dtype=wp.float32),
    last_air_time: wp.array2d(dtype=wp.float32),
    last_contact_time: wp.array2d(dtype=wp.float32),
):
    """Update the net forces, force matrix and air/contact time for each (env, sensor) pair.

    Launch with dim=(num_envs, num_sensors).

    Args:
        net_forces_flat: Flat net forces. Shape is (num_envs*num_sensors,).
        net_forces_matrix_flat: Flat force matrix. Shape is (num_envs*num_sensors, num_filter_shapes).
        mask: Mask array. Shape is (num_envs,).
        num_sensors: Number of sensors per environment.
        num_filter_shapes: Number of filter shapes.
        history_length: Length of history.
        contact_force_threshold: Threshold for the contact force.
        timestamp: Timestamp array. Shape is (num_envs,).
        timestamp_last_update: Timestamp last update array. Shape is (num_envs,).
        net_forces_w: Net forces array. Shape is (num_envs, num_sensors).
        net_forces_w_history: Net forces history array. Shape is (num_envs, history_length, num_sensors).
        force_matrix_w: Force matrix array. Shape is (num_envs, num_sensors, num_filter_shapes).
        force_matrix_w_history: Force matrix history array. Shape is
            (num_envs, history_length, num_sensors, num_filter_shapes).
        current_air_time: Current air time array. Shape is (num_envs, num_sensors).
        current_contact_time: Current contact time array. Shape is (num_envs, num_sensors).
        last_air_time: Last air time array. Shape is (num_envs, num_sensors).
        last_contact_time: Last contact time array. Shape is (num_envs, num_sensors).
    """
    env, sensor = wp.tid()

    if mask:
        if not mask[env]:
            return

    src_idx = env * num_sensors + sensor

    # Update net forces
    net_forces_w[env, sensor] = net_forces_flat[src_idx]
    # Update history
    if net_forces_w_history:
        for i in range(history_length - 1, 0, -1):
            net_forces_w_history[env, i, sensor] = net_forces_w_history[env, i - 1, sensor]
        net_forces_w_history[env, 0, sensor] = net_forces_w[env, sensor]

    # update force matrix
    if net_forces_matrix_flat:
        for f in range(num_filter_shapes):
            force_matrix_w[env, sensor, f] = net_forces_matrix_flat[src_idx, f]
            for i in range(history_length - 1, 0, -1):
                force_matrix_w_history[env, i, sensor, f] = force_matrix_w_history[env, i - 1, sensor, f]
            force_matrix_w_history[env, 0, sensor, f] = force_matrix_w[env, sensor, f]

    # Update air/contact time tracking
    if current_air_time:
        elapsed_time = timestamp[env] - timestamp_last_update[env]
        in_contact = wp.length_sq(net_forces_w[env, sensor]) > contact_force_threshold * contact_force_threshold

        cat = current_air_time[env, sensor]
        cct = current_contact_time[env, sensor]
        is_first_contact = in_contact and (cat > 0.0)
        is_first_detached = not in_contact and (cct > 0.0)

        if is_first_contact:
            last_air_time[env, sensor] = cat + elapsed_time
        elif is_first_detached:
            last_contact_time[env, sensor] = cct + elapsed_time

        current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)
        current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)


@wp.kernel
def concat_pos_and_quat_to_pose_kernel(
    pos: wp.array2d(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    pose: wp.array2d(dtype=wp.transformf),
):
    """Concatenate position and quaternion to pose.

    Args:
        pos: Position array. Shape is (N, B).
        quat: Quaternion array. Shape is (N, B).
        pose: Pose array. Shape is (N, B).
    """
    env, sensor = wp.tid()
    pose[env, sensor] = wp.transform(pos[env, sensor], quat[env, sensor])
