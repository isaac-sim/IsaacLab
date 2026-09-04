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
    num_sensors: int,
    newton_total_force: wp.array(dtype=wp.vec3f),  # (n_envs * n_sensors)
    newton_force_matrix: wp.array2d(dtype=wp.vec3f),  # (n_envs * n_sensors, n_filter_objects) or None
    timestamp: wp.array(dtype=wp.float32),
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

    # Skip envs that have not been stepped since their last reset: Newton's contact buffer
    # still holds pre-reset values, so reading it now would inject stale data (#4970).
    if timestamp[env] == 0.0:
        return

    # Copy total force (column 0) - only thread with f_idx == 0 does this
    src_idx = env * num_sensors + sensor
    if f_idx == 0:
        net_force_total[env, sensor] = newton_total_force[src_idx]

    # Copy per-filter-object forces.
    # Guard with `if force_matrix:` to handle None case (no filter objects)
    if force_matrix:
        force_matrix[env, sensor, f_idx] = newton_force_matrix[src_idx, f_idx]


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
    first_contact_latch: wp.array2d(dtype=wp.float32),
    first_air_latch: wp.array2d(dtype=wp.float32),
    first_contact_time: wp.array2d(dtype=wp.float32),
    first_air_time: wp.array2d(dtype=wp.float32),
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
        first_contact_latch[env, sensor] = 0.0
        first_air_latch[env, sensor] = 0.0
        first_contact_time[env, sensor] = 0.0
        first_air_time[env, sensor] = 0.0


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
    first_contact_latch: wp.array2d(dtype=wp.float32),
    first_air_latch: wp.array2d(dtype=wp.float32),
    first_contact_time: wp.array2d(dtype=wp.float32),
    first_air_time: wp.array2d(dtype=wp.float32),
):
    """Update the contact sensor data (history and air/contact time tracking).

    The transition latches (``first_contact_latch`` / ``first_air_latch``)
    record that a touchdown / lift-off happened, together with the phase age
    since the transition (measured from the midpoint of the transition
    interval). They persist across sensor updates until the phase ends, so a
    first contact / first air event can no longer be missed because of float32
    clock rounding in the timer arithmetic (issue #7283).

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

        # Latch first-contact / first-air events so they cannot be missed
        # because of float32 clock rounding in the timer arithmetic (#7283).
        # The phase age is counted from the midpoint of the transition interval,
        # which leaves a interval/2 margin on both sides of the comparison
        # against dt in compute_first_transition_kernel.
        fct = first_contact_time[env, sensor]
        fat = first_air_time[env, sensor]
        if is_first_contact:
            first_contact_latch[env, sensor] = 1.0
            first_contact_time[env, sensor] = 0.5 * elapsed_time
        elif in_contact:
            first_contact_time[env, sensor] = fct + elapsed_time
        else:
            first_contact_latch[env, sensor] = 0.0
            first_contact_time[env, sensor] = 0.0
        if is_first_detached:
            first_air_latch[env, sensor] = 1.0
            first_air_time[env, sensor] = 0.5 * elapsed_time
        elif not in_contact:
            first_air_time[env, sensor] = fat + elapsed_time
        else:
            first_air_latch[env, sensor] = 0.0
            first_air_time[env, sensor] = 0.0


@wp.kernel
def compute_first_transition_kernel(
    # in
    dt: wp.float32,
    abs_tol: wp.float32,
    transition_latch: wp.array2d(dtype=wp.float32),
    transition_time: wp.array2d(dtype=wp.float32),
    # out
    result: wp.array2d(dtype=wp.float32),
):
    """Compute boolean mask (as float) for sensors whose latched transition happened within the last dt.

    Used by both compute_first_contact (with the first-contact latch) and
    compute_first_air (with the first-air latch). The latch is set by the
    update kernel at the transition sample and cleared when the phase ends,
    and ``transition_time`` tracks the phase age from the midpoint of the
    transition interval, so this comparison is robust to the float32 clock
    rounding error that grows with the simulation time (issue #7283).

    Launch with dim=(num_envs, num_sensors).
    """
    env, sensor = wp.tid()
    if transition_latch[env, sensor] > 0.0 and transition_time[env, sensor] < dt + abs_tol:
        result[env, sensor] = 1.0
    else:
        result[env, sensor] = 0.0
