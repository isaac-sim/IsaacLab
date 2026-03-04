# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

import warp as wp


@wp.kernel
def copy_from_newton_kernel(
    # in
    env_mask: wp.array(dtype=wp.bool),
    newton_forces: wp.array3d(dtype=wp.vec3f),  # (n_envs, n_sensors, n_counterparts)
    # outputs
    net_force_total: wp.array2d(dtype=wp.vec3f),  # (n_envs, n_sensors)
    force_matrix: wp.array3d(dtype=wp.vec3f),  # (n_envs, n_sensors, n_filter_objects) or None
):
    """Copy contact force data from Newton sensor into owned buffers.

    Launch with dim=(num_envs, num_sensors, max(num_filter_objects, 1)) for coalescing.
    When num_filter_objects == 0, trailing dim is 1 and only total is copied.
    """
    env, sensor, f_idx = wp.tid()

    if env_mask:
        if not env_mask[env]:
            return

    # Copy total force (column 0) - only thread with f_idx == 0 does this
    if f_idx == 0:
        net_force_total[env, sensor] = newton_forces[env, sensor, 0]

    # Copy per-filter-object forces (columns 1+)
    # Guard with `if force_matrix:` to handle None case (no filter objects)
    if force_matrix:
        force_matrix[env, sensor, f_idx] = newton_forces[env, sensor, f_idx + 1]


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
):
    """Reset the contact sensor data for specified environments.

    Launch with dim=(num_envs, num_sensors).
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


@wp.kernel
def update_contact_sensor_kernel(
    # in
    history_length: int,
    contact_force_threshold: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    net_forces: wp.array2d(dtype=wp.vec3f),
    timestamp: wp.array(dtype=wp.float32),
    timestamp_last_update: wp.array(dtype=wp.float32),
    # in-out
    net_forces_history: wp.array3d(dtype=wp.vec3f),
    current_air_time: wp.array2d(dtype=wp.float32),
    current_contact_time: wp.array2d(dtype=wp.float32),
    # out
    last_air_time: wp.array2d(dtype=wp.float32),
    last_contact_time: wp.array2d(dtype=wp.float32),
):
    """Update the contact sensor data (history and air/contact time tracking).

    Launch with dim=(num_envs, num_sensors).
    """
    env, sensor = wp.tid()

    if env_mask:
        if not env_mask[env]:
            return

    # Update history
    if net_forces_history:
        for i in range(history_length - 1, 0, -1):
            net_forces_history[env, i, sensor] = net_forces_history[env, i - 1, sensor]
        net_forces_history[env, 0, sensor] = net_forces[env, sensor]

    # Update air/contact time tracking
    if current_air_time:
        elapsed_time = timestamp[env] - timestamp_last_update[env]
        in_contact = wp.length_sq(net_forces[env, sensor]) > contact_force_threshold * contact_force_threshold

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
    """
    env, sensor = wp.tid()
    t = time[env, sensor]
    if t > 0.0 and t < threshold:
        result[env, sensor] = 1.0
    else:
        result[env, sensor] = 0.0
