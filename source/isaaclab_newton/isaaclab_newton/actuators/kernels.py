# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for the Newton actuator fast path."""

import warp as wp


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


