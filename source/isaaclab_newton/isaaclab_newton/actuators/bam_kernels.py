# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the BAM part Newton does not provide: the gearbox friction."""

from __future__ import annotations

import warp as wp


@wp.kernel
def bam_friction_budget_kernel(
    joint_qd: wp.array[float],
    dof_indices: wp.array[wp.uint32],
    external_torque: wp.array[float],
    prev_applied_torque: wp.array[float],
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
    stribeck: int,
    load_dependent: int,
    quadratic: int,
    friction_budget: wp.array[float],
):
    """Compute the velocity-independent gearbox friction budget from the servo velocity and transmitted loads."""
    i = wp.tid()
    joint_vel = joint_qd[dof_indices[i]]
    ext_tau = external_torque[i]

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
def zero_masked_envs_kernel(
    env_mask: wp.array[wp.bool],
    dof_indices: wp.array[wp.uint32],
    dofs_per_env: int,
    values: wp.array[float],
):
    """Zero the slots whose DOF belongs to a masked environment, or every slot without a mask."""
    i = wp.tid()
    if env_mask:
        if not env_mask[int(dof_indices[i]) // dofs_per_env]:
            return
    values[i] = 0.0
