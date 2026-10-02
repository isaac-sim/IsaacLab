# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for BAM command delay, motor control, gearbox friction, and state reset."""

from __future__ import annotations

import warp as wp


@wp.kernel
def _bam_delay_kernel(
    target_pos: wp.array[float],
    target_pos_indices: wp.array[wp.uint32],
    delay_ring: wp.array2d[float],
    delay_lag: wp.array[wp.int32],
    delay_fill: wp.array[wp.int32],
    delay_step_count: wp.array[wp.int32],
    delay_phase: wp.array[wp.int32],
    min_delay: int,
    max_delay: int,
    delay_hold_prob: float,
    delay_update_period: int,
    delay_seed: wp.array[wp.int32],
    delayed_target: wp.array[float],
    next_ring: wp.array2d[float],
    next_lag: wp.array[wp.int32],
    next_fill: wp.array[wp.int32],
    next_step_count: wp.array[wp.int32],
):
    """Draw the command lag, read the delayed target, and advance its history."""
    i = wp.tid()

    target = target_pos[target_pos_indices[i]]
    step_count = delay_step_count[i]
    # Reset clears the cached lag; a held or staggered update must still honor the minimum.
    lag = wp.max(delay_lag[i], min_delay)
    fill = delay_fill[i]

    if max_delay > 0:
        # Resample the lag before reading, matching the reference update policy:
        # a draw is attempted only on the environment's phase of the update period and
        # may be skipped again with probability ``delay_hold_prob``.
        should_update = True
        if delay_update_period > 0:
            should_update = ((step_count + delay_phase[i]) % delay_update_period) == 0
        rng = wp.rand_init(delay_seed[i], step_count)
        if should_update:
            if delay_hold_prob > 0.0:
                should_update = wp.randf(rng) >= delay_hold_prob
        if should_update:
            lag = wp.randi(rng, min_delay, max_delay + 1)

        # ``delay_ring[i, 0]`` is the previous step's command, so a lag of ``k`` reads
        # column ``k - 1``. The read is clamped to the number of commands seen so far,
        # which is how the reference ring behaves before it has filled up.
        if lag > 0 and fill > 0:
            column = wp.min(lag - 1, fill - 1)
            target = delay_ring[i, column]

        next_ring[i, 0] = target_pos[target_pos_indices[i]]
        for column in range(1, max_delay):
            next_ring[i, column] = delay_ring[i, column - 1]
        next_fill[i] = wp.min(fill + 1, max_delay)
    else:
        next_fill[i] = 0
    next_lag[i] = lag
    next_step_count[i] = step_count + 1
    delayed_target[i] = target


@wp.kernel
def _bam_motor_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    delayed_target: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    backlash_pos_indices: wp.array[wp.uint32],
    kp_fw: wp.array[float],
    kp_scale: wp.array[float],
    kd_scale: wp.array[float],
    vin: wp.array[float],
    sag_gain: wp.array[float],
    kt: wp.array[float],
    resistance: wp.array[float],
    error_gain: wp.array[float],
    max_pwm: wp.array[float],
    max_current: wp.array[float],
    prev_motor_torque: wp.array[float],
    vin_min: float,
    env_dof_stride: int,
    motor_torque: wp.array[float],
    effective_vin: wp.array[float],
):
    """Sag the supply and convert the delayed command through firmware PWM to motor torque."""
    i = wp.tid()

    # All joints sharing one supply sag together, so the drop is driven by the summed
    # magnitude of the torques the environment's joints drew on the previous step.
    load = float(0.0)
    base = i - (i % env_dof_stride)
    for offset in range(env_dof_stride):
        load += wp.abs(prev_motor_torque[base + offset])
    vin_eff = vin[i] - sag_gain[i] * load
    vin_eff = wp.max(vin_eff, vin_min)
    effective_vin[i] = vin_eff

    # kd_scale scales the electrical damping only: the velocity enters the firmware law
    # and the torque equation solely through the back-EMF term.
    scaled_vel = velocities[vel_indices[i]] * kd_scale[i]

    measured = positions[pos_indices[i]]
    if backlash_pos_indices:
        measured += positions[backlash_pos_indices[i]]

    duty = (delayed_target[i] - measured) * (kp_fw[i] * kp_scale[i]) * error_gain[i]
    if max_current[i] > 0.0:
        duty_center = kt[i] * scaled_vel / vin_eff
        duty_span = resistance[i] * max_current[i] / vin_eff
        duty = wp.clamp(duty, duty_center - duty_span, duty_center + duty_span)
    duty = wp.clamp(duty, -max_pwm[i], max_pwm[i])

    volts = vin_eff * duty
    motor_torque[i] = kt[i] * volts / resistance[i] - (kt[i] * kt[i]) * scaled_vel / resistance[i]


@wp.kernel
def _bam_friction_kernel(
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    motor_torque: wp.array[float],
    external_torque_in: wp.array[float],
    friction_scale: wp.array[float],
    friction_base: wp.array[float],
    friction_stribeck: wp.array[float],
    dtheta_stribeck: wp.array[float],
    alpha: wp.array[float],
    load_friction_motor: wp.array[float],
    load_friction_external: wp.array[float],
    load_friction_motor_stribeck: wp.array[float],
    load_friction_external_stribeck: wp.array[float],
    load_friction_motor_quad: wp.array[float],
    load_friction_external_quad: wp.array[float],
    max_effort: wp.array[float],
    prev_applied_torque: wp.array[float],
    stribeck: int,
    load_dependent: int,
    quadratic: int,
    forces: wp.array[float],
    friction_budget: wp.array[float],
    next_prev_motor: wp.array[float],
    next_prev_applied: wp.array[float],
):
    """Publish the friction budget for MJWarp and emit the clamped motor torque."""
    i = wp.tid()
    joint_vel = velocities[vel_indices[i]]
    motor_tau = motor_torque[i]
    ext_tau = external_torque_in[i]

    stribeck_coeff = float(0.0)
    if stribeck != 0:
        stribeck_coeff = wp.exp(-wp.pow(wp.abs(joint_vel) / dtheta_stribeck[i], alpha[i]))

    prev_tau = prev_applied_torque[i]
    budget = friction_base[i]
    if stribeck != 0:
        budget += stribeck_coeff * friction_stribeck[i]
    if load_dependent != 0:
        budget += wp.abs(ext_tau * load_friction_external[i] - prev_tau * load_friction_motor[i])
        if stribeck != 0:
            budget += stribeck_coeff * wp.abs(
                ext_tau * load_friction_external_stribeck[i] - prev_tau * load_friction_motor_stribeck[i]
            )
            if quadratic != 0:
                # Driving (motor wins) loads the gearbox through the external torque;
                # back-driving (load wins) loads it through the motor torque.
                abs_ext = wp.abs(ext_tau)
                abs_motor = wp.abs(prev_tau)
                quad_term = load_friction_motor_quad[i] * abs_motor * abs_motor
                if abs_motor > abs_ext:
                    quad_term = load_friction_external_quad[i] * abs_ext * abs_ext
                budget += stribeck_coeff * quad_term
    budget *= friction_scale[i]

    friction_budget[i] = budget

    # BAM owns its effort clamp so a registered clamping schema cannot hide its
    # unregistered controller token from Newton's USD component discovery.
    forces[i] = wp.clamp(motor_tau, -max_effort[i], max_effort[i])

    next_prev_motor[i] = motor_tau
    next_prev_applied[i] = forces[i]


@wp.kernel
def _bam_state_reset_kernel(
    mask: wp.array[wp.bool],
    prev_motor_torque: wp.array[float],
    prev_applied_torque: wp.array[float],
    delay_ring: wp.array2d[float],
    delay_lag: wp.array[wp.int32],
    delay_fill: wp.array[wp.int32],
    delay_step_count: wp.array[wp.int32],
    delay_phase: wp.array[wp.int32],
    delay_rng_seed: wp.array[wp.int32],
    delay_update_period: int,
    env_dof_stride: int,
    reset_seed: int,
):
    """Clear the previous-step caches and the delay state of the masked DOFs."""
    i = wp.tid()
    if mask:
        if not mask[i]:
            return
    prev_motor_torque[i] = 0.0
    prev_applied_torque[i] = 0.0
    delay_lag[i] = 0
    delay_fill[i] = 0
    delay_step_count[i] = 0
    delay_rng_seed[i] = wp.int32(wp.rand_init(reset_seed, i // env_dof_stride))
    for column in range(delay_ring.shape[1]):
        delay_ring[i, column] = 0.0
    if delay_update_period > 0:
        delay_phase[i] = wp.randi(wp.rand_init(reset_seed, i // env_dof_stride), 0, delay_update_period)
    else:
        delay_phase[i] = 0
