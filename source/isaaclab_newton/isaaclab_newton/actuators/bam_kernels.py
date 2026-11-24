# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the BAM parts Newton does not provide: command lag and gearbox friction."""

from __future__ import annotations

import warp as wp


@wp.kernel
def bam_command_lag_kernel(
    lag: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    rng_seed: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    min_delay: int,
    max_delay: int,
    hold_prob: float,
    update_period: int,
    delay_steps: wp.array[wp.int32],
):
    """Draw this step's command lag and write it into the delay processor's slots."""
    i = wp.tid()
    count = step_count[i]
    # Reset clears the cached lag; a held or staggered update must still honor the minimum.
    current = wp.max(lag[i], min_delay)
    # A draw is attempted only on the environment's phase of the update period and may be
    # skipped again with probability ``hold_prob``.
    should_update = True
    if update_period > 0:
        should_update = ((count + phase[i]) % update_period) == 0
    rng = wp.rand_init(rng_seed[i], count)
    if should_update:
        if hold_prob > 0.0:
            should_update = wp.randf(rng) >= hold_prob
    if should_update:
        current = wp.randi(rng, min_delay, max_delay + 1)
    lag[i] = current
    step_count[i] = count + 1
    delay_steps[slots[i]] = current


@wp.kernel
def bam_command_lag_reset_kernel(
    env_mask: wp.array[wp.bool],
    dof_indices: wp.array[wp.uint32],
    dofs_per_env: int,
    first_dof: int,
    update_period: int,
    reset_seed: int,
    lag: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    rng_seed: wp.array[wp.int32],
):
    """Restart the lag stream of the masked environments from an episode seed shared by each environment's DOFs.

    The stream is keyed by the DOF of the lag's first servo in the environment, ``first_dof`` within each
    environment, so the lags of other groups and environments draw independently.
    """
    i = wp.tid()
    env = int(dof_indices[i]) // dofs_per_env
    if env_mask:
        if not env_mask[env]:
            return
    lag[i] = 0
    step_count[i] = 0
    stream = wp.rand_init(reset_seed, env * dofs_per_env + first_dof)
    rng_seed[i] = wp.int32(stream)
    if update_period > 0:
        phase[i] = wp.randi(stream, 0, update_period)
    else:
        phase[i] = 0


@wp.struct
class BamFrictionFit:
    """Gearbox friction fit of a BAM motor, as in :class:`~isaaclab.actuators.BamMotorCfg`.

    The flags select the terms of the fit's model, as in :data:`~isaaclab.actuators.actuator_bam_cfg.BAM_MODEL_FLAGS`.
    """

    friction_base: float
    friction_stribeck: float
    dtheta_stribeck: float
    alpha: float
    load_friction_motor: float
    load_friction_external: float
    load_friction_motor_stribeck: float
    load_friction_external_stribeck: float
    load_friction_motor_quad: float
    load_friction_external_quad: float
    stribeck: int
    load_dependent: int
    quadratic: int


@wp.kernel
def bam_friction_budget_kernel(
    joint_qd: wp.array[float],
    dof_indices: wp.array[wp.uint32],
    qfrc_bias: wp.array[float],
    qfrc_constraint: wp.array[float],
    qfrc_frictionloss: wp.array[float],
    prev_applied_torque: wp.array[float],
    episode_start: wp.array[wp.bool],
    friction_scale: wp.array[float],
    fit: BamFrictionFit,
    external_torque: wp.array[float],
    friction_budget: wp.array[float],
    joint_friction: wp.array[float],
):
    """Compute the velocity-independent gearbox friction budget from the servo velocity and transmitted loads.

    The external load is the previous solve's bias and constraint forces without the servo's own friction loss,
    which would otherwise feed the load-dependent terms back on themselves. The first budget of an episode reads
    no external load, since the previous solve belongs to the terminated episode.
    """
    i = wp.tid()
    dof = dof_indices[i]
    joint_vel = joint_qd[dof]
    # Copy the flag before clearing it: a branch on episode_start[i] would read the array again after the store.
    starts_episode = episode_start[i]
    episode_start[i] = False
    ext_tau = float(0.0)
    if not starts_episode:
        ext_tau = -qfrc_bias[dof] + qfrc_constraint[dof] - qfrc_frictionloss[dof]
    external_torque[i] = ext_tau

    stribeck_coeff = float(0.0)
    if fit.stribeck != 0:
        stribeck_coeff = wp.exp(-wp.pow(wp.abs(joint_vel) / fit.dtheta_stribeck, fit.alpha))

    prev_tau = prev_applied_torque[i]
    budget = fit.friction_base
    if fit.stribeck != 0:
        budget += stribeck_coeff * fit.friction_stribeck
    if fit.load_dependent != 0:
        budget += wp.abs(ext_tau * fit.load_friction_external - prev_tau * fit.load_friction_motor)
        if fit.stribeck != 0:
            budget += stribeck_coeff * wp.abs(
                ext_tau * fit.load_friction_external_stribeck - prev_tau * fit.load_friction_motor_stribeck
            )
            if fit.quadratic != 0:
                # Driving (motor wins) loads the gearbox through the external torque;
                # back-driving (load wins) loads it through the motor torque.
                abs_ext = wp.abs(ext_tau)
                abs_motor = wp.abs(prev_tau)
                quad_term = fit.load_friction_motor_quad * abs_motor * abs_motor
                if abs_motor > abs_ext:
                    quad_term = fit.load_friction_external_quad * abs_ext * abs_ext
                budget += stribeck_coeff * quad_term
    budget *= friction_scale[i]

    friction_budget[i] = budget
    joint_friction[dof] = budget


@wp.kernel
def gather_dof_values_kernel(
    values: wp.array[float],
    dof_indices: wp.array[wp.uint32],
    out: wp.array[float],
):
    """Copy DOF-indexed values into per-slot storage."""
    i = wp.tid()
    out[i] = values[dof_indices[i]]


@wp.kernel
def bam_friction_reset_kernel(
    env_mask: wp.array[wp.bool],
    dof_indices: wp.array[wp.uint32],
    dofs_per_env: int,
    prev_applied_torque: wp.array[float],
    episode_start: wp.array[wp.bool],
):
    """Start a new episode on the slots whose DOF belongs to a masked environment, or on every slot without a mask."""
    i = wp.tid()
    if env_mask:
        if not env_mask[int(dof_indices[i]) // dofs_per_env]:
            return
    prev_applied_torque[i] = 0.0
    episode_start[i] = True
