# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for the Newton actuator fast path."""

import warp as wp


@wp.kernel
def combine_actuation_force(
    user_ff: wp.array2d(dtype=wp.float32),
    newton_output: wp.array2d(dtype=wp.float32),
    joint_modes: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    """Per-DOF select between user FF (``mode==1``, implicit) and Newton actuator output (``mode==0``).

    For explicit DOFs the user FF was already folded into
    :paramref:`newton_output` via Newton's ``joint_act``, so taking it
    from there avoids double-counting.

    Args:
        user_ff: User-commanded feedforward effort [N·m or N], shape ``(num_envs, num_joints)``.
        newton_output: Post-clamp Newton actuator output [N·m or N], shape ``(num_envs, num_joints)``.
        joint_modes: Per-DOF mode (``0`` = explicit, ``1`` = implicit), shape ``(num_joints,)``.
        out: Output actuation force buffer [N·m or N], shape ``(num_envs, num_joints)``.
    """
    i, j = wp.tid()
    if joint_modes[j] == 1:
        out[i, j] = user_ff[i, j]
    else:
        out[i, j] = newton_output[i, j]


@wp.kernel
def compute_actuator_telemetry(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
    joint_vel_target: wp.array2d(dtype=wp.float32),
    joint_effort_target: wp.array2d(dtype=wp.float32),
    joint_effort_actual: wp.array2d(dtype=wp.float32),
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    joint_modes: wp.array(dtype=wp.int32),
    target_computed_effort: wp.array2d(dtype=wp.float32),
    target_applied_effort: wp.array2d(dtype=wp.float32),
):
    """Per-DOF actuator torque telemetry.

    For ``mode==0`` (explicit) copies :paramref:`joint_effort_actual` into both outputs.
    For ``mode==1`` (implicit) reproduces :meth:`isaaclab.actuators.ImplicitActuator.compute` —
    ``kp*(q_des-q) + kd*(v_des-v) + ff`` with optional clip — as a shadow computation;
    the simulator runs the real PD.

    Args:
        joint_pos: Current positions [rad or m, depending on joint type], shape ``(num_envs, num_joints)``.
        joint_vel: Current velocities [rad/s or m/s], shape ``(num_envs, num_joints)``.
        joint_pos_target: Position targets [rad or m, depending on joint type], shape ``(num_envs, num_joints)``.
        joint_vel_target: Velocity targets [rad/s or m/s], shape ``(num_envs, num_joints)``.
        joint_effort_target: Feedforward efforts [N·m or N], shape ``(num_envs, num_joints)``.
        joint_effort_actual: Post-step actuator output [N·m or N], shape ``(num_envs, num_joints)``.
        stiffness: Per-joint kp [N·m/rad], shape ``(num_envs, num_joints)``.
        damping: Per-joint kd [N·m·s/rad], shape ``(num_envs, num_joints)``.
        effort_limit: Absolute effort limit [N·m or N]; use ``inf`` to disable clipping. Shape ``(num_envs, num_joints)``.
        joint_indices: DOF indices to process, shape ``(num_actuated_joints,)``.
        joint_modes: Per-entry mode (``0`` = copy, ``1`` = compute), shape ``(num_actuated_joints,)``.
        target_computed_effort: Output unclipped effort [N·m or N], shape ``(num_envs, num_joints)``.
        target_applied_effort: Output post-clip effort [N·m or N], shape ``(num_envs, num_joints)``.
    """
    i, j = wp.tid()
    dof = joint_indices[j]
    if joint_modes[j] == 0:
        actual = joint_effort_actual[i, dof]
        target_computed_effort[i, dof] = actual
        target_applied_effort[i, dof] = actual
    else:
        err_p = joint_pos_target[i, dof] - joint_pos[i, dof]
        err_v = joint_vel_target[i, dof] - joint_vel[i, dof]
        computed = stiffness[i, dof] * err_p + damping[i, dof] * err_v + joint_effort_target[i, dof]
        target_computed_effort[i, dof] = computed
        limit = effort_limit[i, dof]
        target_applied_effort[i, dof] = wp.clamp(computed, -limit, limit)
