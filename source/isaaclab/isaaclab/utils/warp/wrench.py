# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused wrench algebra and selected reset for :class:`~isaaclab.utils.wrench_composer.WrenchComposer`.

The composer owns buffers and chooses their frame at launch time. Kernels consume device-resident
selections, so their contents may change during graph replay. Factories create only requested
selector/operation specializations in isolated modules; warm the actual call path before capture.
Each selected destination has one writer. Use ordinary read-add-write assignments rather than
array ``+=``, which Warp may lower to atomics intended for concurrent reduction.
"""

from functools import cache

import warp as wp


@wp.func
def _set_wrench(
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    force_buffer: wp.array2d(dtype=wp.vec3f),
    torque_buffer: wp.array2d(dtype=wp.vec3f),
    i: int,
    j: int,
    env: int,
    body: int,
):
    if torques:
        torque_buffer[env, body] = torques[i, j]
    if forces:
        force_buffer[env, body] = forces[i, j]
        if positions:
            moment = wp.cross(positions[i, j], forces[i, j])
            if torques:
                torque_buffer[env, body] = torque_buffer[env, body] + moment
            else:
                torque_buffer[env, body] = moment


@wp.func
def _add_wrench(
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    force_buffer: wp.array2d(dtype=wp.vec3f),
    torque_buffer: wp.array2d(dtype=wp.vec3f),
    i: int,
    j: int,
    env: int,
    body: int,
):
    if forces:
        force_buffer[env, body] = force_buffer[env, body] + forces[i, j]
        if positions:
            torque_buffer[env, body] = torque_buffer[env, body] + wp.cross(positions[i, j], forces[i, j])
    if torques:
        torque_buffer[env, body] = torque_buffer[env, body] + torques[i, j]


@wp.func
def _reset_wrench(
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
    env: int,
    body: int,
):
    z = wp.vec3f(0.0)
    global_force_w[env, body] = z
    global_torque_w[env, body] = z
    global_force_at_com_w[env, body] = z
    local_force_b[env, body] = z
    local_torque_b[env, body] = z
    out_force_b[env, body] = z
    out_torque_b[env, body] = z


@cache
def make_write_wrench(env_dtype=wp.bool, body_dtype=wp.bool, *, accumulate: bool = False) -> wp.Kernel:
    """Return a fused force, torque and moment writer for the requested selectors.

    Args:
        env_dtype: ``wp.bool`` for masks, or ``wp.int32``/``wp.int64`` for indices.
        body_dtype: Boolean for masked selection, otherwise either signed index width.
        accumulate: Add contributions when True; assign provided contributions otherwise.

    Returns:
        A cached concrete kernel. Indexed launches use compact inputs and
        ``dim=(len(env_selection), len(body_selection))``; masked launches use full-shaped
        inputs and ``dim=(num_envs, num_bodies)``. Index lists must contain unique, valid entries.

        Forces [N], torques [N m] and positions [m] use the frame selected by the caller's
        destination buffers. World positions are absolute; body positions are offsets from
        the center of mass. Positional forces add ``cross(position, force)`` to torque; the
        composition kernel later subtracts the world center-of-mass moment. Null input arrays
        leave their contributions unchanged, and positions without forces have no effect.
        The caller owns clearing existing contributions before replacement.
    """
    indexed = env_dtype != wp.bool
    if indexed:
        if env_dtype not in (wp.int32, wp.int64) or body_dtype not in (wp.int32, wp.int64):
            raise TypeError("Indexed wrench writes require signed integer selectors")
    elif body_dtype != wp.bool:
        raise TypeError("Masked wrench writes require boolean selectors")

    @wp.kernel(module="unique")
    def write_wrench(
        env_selection: wp.array(dtype=env_dtype),
        body_selection: wp.array(dtype=body_dtype),
        forces: wp.array2d(dtype=wp.vec3f),
        torques: wp.array2d(dtype=wp.vec3f),
        positions: wp.array2d(dtype=wp.vec3f),
        force_buffer: wp.array2d(dtype=wp.vec3f),
        torque_buffer: wp.array2d(dtype=wp.vec3f),
    ):
        i, j = wp.tid()
        if wp.static(indexed):
            env = wp.int32(env_selection[i])
            body = wp.int32(body_selection[j])
        else:
            if not env_selection[i] or not body_selection[j]:
                return
            env = i
            body = j

        if wp.static(accumulate):
            _add_wrench(forces, torques, positions, force_buffer, torque_buffer, i, j, env, body)
        else:
            _set_wrench(forces, torques, positions, force_buffer, torque_buffer, i, j, env, body)

    return write_wrench


@wp.kernel
def add_raw_wrench_buffers(
    src_global_force_w: wp.array2d(dtype=wp.vec3f),
    src_global_torque_w: wp.array2d(dtype=wp.vec3f),
    src_global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    src_local_force_b: wp.array2d(dtype=wp.vec3f),
    src_local_torque_b: wp.array2d(dtype=wp.vec3f),
    dst_global_force_w: wp.array2d(dtype=wp.vec3f),
    dst_global_torque_w: wp.array2d(dtype=wp.vec3f),
    dst_global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    dst_local_force_b: wp.array2d(dtype=wp.vec3f),
    dst_local_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Element-wise add all five source wrench buffers into destination buffers.

    Dispatched with ``dim=(num_envs, num_bodies)``. Each ``src_*`` / ``dst_*`` pair corresponds
    to one input buffer, preserving its world (``_w``) or body (``_b``) frame. Forces are in N,
    torques in N m; positional global forces retain their world-origin moment convention.
    """
    i, j = wp.tid()
    dst_global_force_w[i, j] = dst_global_force_w[i, j] + src_global_force_w[i, j]
    dst_global_torque_w[i, j] = dst_global_torque_w[i, j] + src_global_torque_w[i, j]
    dst_global_force_at_com_w[i, j] = dst_global_force_at_com_w[i, j] + src_global_force_at_com_w[i, j]
    dst_local_force_b[i, j] = dst_local_force_b[i, j] + src_local_force_b[i, j]
    dst_local_torque_b[i, j] = dst_local_torque_b[i, j] + src_local_torque_b[i, j]


@wp.kernel
def compose_wrench_to_body_frame(
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    com_pos_w: wp.array2d(dtype=wp.vec3f),
    link_quat_w: wp.array2d(dtype=wp.quatf),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Compose global and local wrench buffers into a single body-frame output.

    Global torques store the moment of positional forces about the world origin: ``cross(P, F)``.
    This kernel corrects to be about the body's CoM via ``cross(P, F) - cross(com_pos_w, F) =
    cross(P - com_pos_w, F)``, then rotates both force and torque into the body frame using
    ``quat_rotate_inv(link_quat_w, ...)``, and adds local-frame values.

    Dispatched with ``dim=(num_envs, num_bodies)``.
    """
    tid_env, tid_body = wp.tid()
    total_force_w = global_force_w[tid_env, tid_body] + global_force_at_com_w[tid_env, tid_body]
    corrected_torque_w = global_torque_w[tid_env, tid_body] - wp.cross(
        com_pos_w[tid_env, tid_body], global_force_w[tid_env, tid_body]
    )
    out_force_b[tid_env, tid_body] = (
        wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], total_force_w) + local_force_b[tid_env, tid_body]
    )
    out_torque_b[tid_env, tid_body] = (
        wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], corrected_torque_w) + local_torque_b[tid_env, tid_body]
    )


@cache
def make_reset_wrench(dtype=wp.bool) -> wp.Kernel:
    """Return a fused reset for a boolean mask or signed integer selection.

    Launch with ``dim=(len(selection), num_bodies)``. Masks span every environment; index
    lists contain valid environment IDs. All seven buffers are cleared for selected rows.
    """
    if dtype not in (wp.bool, wp.int32, wp.int64):
        raise TypeError("Wrench reset requires boolean or signed integer selectors")

    @wp.kernel(module="unique")
    def reset(
        selection: wp.array(dtype=dtype),
        global_force_w: wp.array2d(dtype=wp.vec3f),
        global_torque_w: wp.array2d(dtype=wp.vec3f),
        global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
        local_force_b: wp.array2d(dtype=wp.vec3f),
        local_torque_b: wp.array2d(dtype=wp.vec3f),
        out_force_b: wp.array2d(dtype=wp.vec3f),
        out_torque_b: wp.array2d(dtype=wp.vec3f),
    ):
        tid_env, tid_body = wp.tid()
        if wp.static(dtype == wp.bool):
            if not selection[tid_env]:
                return
            ei = tid_env
        else:
            ei = wp.int32(selection[tid_env])
        _reset_wrench(
            global_force_w,
            global_torque_w,
            global_force_at_com_w,
            local_force_b,
            local_torque_b,
            out_force_b,
            out_torque_b,
            ei,
            tid_body,
        )

    return reset
