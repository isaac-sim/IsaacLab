# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for the Newton actuator fast path."""

import warp as wp


# ---------------------------------------------------------------------------
# Adapter-internal kernels: per-DOF zeroing, env-mask building, and gain
# scatter/gather between the adapter's flat per-env-per-DOF buffer and the
# per-actuator controller arrays. Used by :class:`NewtonActuatorAdapter`
# (stepping, finalize, gain DR) and by the kernel-based propagator.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def zero_at_indices_kernel(data: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.uint32)):
    """Zero a flat ``data`` buffer at the given flat ``indices``."""
    i = wp.tid()
    data[indices[i]] = 0.0


@wp.kernel(enable_backward=False)
def set_mask_kernel(mask: wp.array(dtype=wp.bool), indices: wp.array(dtype=wp.int32)):
    """Set ``mask[indices[i]] = True`` for each ``i``. The mask must be pre-zeroed."""
    i = wp.tid()
    mask[indices[i]] = True


@wp.kernel(enable_backward=False)
def scatter_gain_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    dof_offset: int,
    num_joints: int,
):
    """Scatter per-actuator ``src`` values into the adapter's flat per-env-per-DOF ``dst``."""
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    local_dof = global_dof % num_joints
    dst[env * num_joints + local_dof] = src[i]


@wp.kernel(enable_backward=False)
def gather_gain_kernel(
    flat_src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    env_mask: wp.array(dtype=wp.bool),
    dof_offset: int,
    num_joints: int,
):
    """Gather from the adapter's flat ``(num_envs * num_joints)`` layout into a
    per-actuator controller array, only for envs where ``env_mask`` is ``True``.
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    if env_mask[env]:
        local_dof = global_dof % num_joints
        dst[i] = flat_src[env * num_joints + local_dof]


@wp.kernel(enable_backward=False)
def scatter_gain_at_envs_kernel(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Scatter ``in_data[i, j]`` into ``out_data[env_ids[i], j]`` for all (i, j)."""
    i, j = wp.tid()
    out_data[env_ids[i], j] = in_data[i, j]


@wp.kernel(enable_backward=False)
def fill_gain_at_envs_kernel(
    value: float,
    env_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Set ``out_data[env_ids[i], j] = value`` for all (i, j)."""
    i, j = wp.tid()
    out_data[env_ids[i], j] = value


# ---------------------------------------------------------------------------
# Articulation-level kernels: in-graph post-actuator hook.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def synch_torque_and_apply_implicit_feedforwards(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
    joint_vel_target: wp.array2d(dtype=wp.float32),
    joint_effort_target: wp.array2d(dtype=wp.float32),
    joint_stiffness: wp.array2d(dtype=wp.float32),
    joint_damping: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    joint_modes: wp.array(dtype=wp.int32),
    sim_bind_joint_effort: wp.array2d(dtype=wp.float32),
    computed: wp.array2d(dtype=wp.float32),
    applied: wp.array2d(dtype=wp.float32),
):
    """In-graph post-actuator hook: route implicit FF and sync telemetry.

    For each (env, dof):
      * Implicit DOF: write user FF to ``joint_f`` (sim integrates it
        alongside the joint-drive PD), clamp the shadow PD ``kp*err_p +
        kd*err_v`` to ``±effort_limit``, and write ``computed = applied
        = PD_clipped + FF``.
      * Explicit DOF: mirror Newton's post-actuator ``joint_f`` into
        ``computed`` / ``applied``.

    Limitation: ``effort_limit`` here only clamps the **PD shadow** used
    for telemetry. The simulator's joint drive applies its max-force
    only to the PD term, so user feedforward effort can exceed
    ``effort_limit`` once it lands in ``joint_f``. Limiting the *total*
    applied effort would require Newton's motor-actuator path
    (configurable via the Newton team), which may have negative perf
    implications.
    """
    i, j = wp.tid()
    if joint_modes[j] == 1:
        sim_bind_joint_effort[i, j] = joint_effort_target[i, j]
        err_p = joint_pos_target[i, j] - joint_pos[i, j]
        err_v = joint_vel_target[i, j] - joint_vel[i, j]
        pd = joint_stiffness[i, j] * err_p + joint_damping[i, j] * err_v
        limit = effort_limit[i, j]
        pd_clipped = wp.clamp(pd, -limit, limit)
        total = pd_clipped + sim_bind_joint_effort[i, j]
        computed[i, j] = total
        applied[i, j] = total
    else:
        val = sim_bind_joint_effort[i, j]
        computed[i, j] = val
        applied[i, j] = val


