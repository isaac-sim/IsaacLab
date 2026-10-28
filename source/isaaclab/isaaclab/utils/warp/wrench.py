# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels owned by WrenchComposer; routing and buffer ownership remain on the host."""

from functools import cache

import warp as wp


@cache
def make_write_wrench(env_dtype=wp.bool, body_dtype=wp.bool, *, accumulate: bool = False) -> wp.Kernel:
    """Generate a writer; selector layout and write operation are compile-time choices."""
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
            if forces:
                force_buffer[env, body] = force_buffer[env, body] + forces[i, j]
                if positions:
                    torque_buffer[env, body] = torque_buffer[env, body] + wp.cross(positions[i, j], forces[i, j])
            if torques:
                torque_buffer[env, body] = torque_buffer[env, body] + torques[i, j]
        else:
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

    return write_wrench


@wp.kernel
def add_raw_wrench_buffers(
    src_gf: wp.array2d(dtype=wp.vec3f),
    src_gt: wp.array2d(dtype=wp.vec3f),
    src_gfc: wp.array2d(dtype=wp.vec3f),
    src_lf: wp.array2d(dtype=wp.vec3f),
    src_lt: wp.array2d(dtype=wp.vec3f),
    dst_gf: wp.array2d(dtype=wp.vec3f),
    dst_gt: wp.array2d(dtype=wp.vec3f),
    dst_gfc: wp.array2d(dtype=wp.vec3f),
    dst_lf: wp.array2d(dtype=wp.vec3f),
    dst_lt: wp.array2d(dtype=wp.vec3f),
):
    """Element-wise add all five source wrench buffers into destination buffers.

    Dispatched with ``dim=(num_envs, num_bodies)``. Each ``src_*`` / ``dst_*`` pair corresponds
    to one of the five input buffers (global_force_w, global_torque_w, global_force_at_com_w,
    local_force_b, local_torque_b).
    """
    tid_env, tid_body = wp.tid()
    dst_gf[tid_env, tid_body] = dst_gf[tid_env, tid_body] + src_gf[tid_env, tid_body]
    dst_gt[tid_env, tid_body] = dst_gt[tid_env, tid_body] + src_gt[tid_env, tid_body]
    dst_gfc[tid_env, tid_body] = dst_gfc[tid_env, tid_body] + src_gfc[tid_env, tid_body]
    dst_lf[tid_env, tid_body] = dst_lf[tid_env, tid_body] + src_lf[tid_env, tid_body]
    dst_lt[tid_env, tid_body] = dst_lt[tid_env, tid_body] + src_lt[tid_env, tid_body]


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
def make_reset_wrench(dtype) -> wp.Kernel:
    """Return a reset kernel for the requested signed index type."""
    if dtype not in (wp.int32, wp.int64):
        raise TypeError("Wrench reset requires signed integer selectors")

    @wp.kernel(module="unique")
    def reset(
        env_ids: wp.array(dtype=dtype),
        global_force_w: wp.array2d(dtype=wp.vec3f),
        global_torque_w: wp.array2d(dtype=wp.vec3f),
        global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
        local_force_b: wp.array2d(dtype=wp.vec3f),
        local_torque_b: wp.array2d(dtype=wp.vec3f),
        out_force_b: wp.array2d(dtype=wp.vec3f),
        out_torque_b: wp.array2d(dtype=wp.vec3f),
    ):
        """Zero all 7 wrench composer buffers at the specified environment indices.

        Dispatched with ``dim=(len(env_ids), num_bodies)``.
        """
        tid_env, tid_body = wp.tid()
        ei = wp.int32(env_ids[tid_env])
        z = wp.vec3f(0.0)
        global_force_w[ei, tid_body] = z
        global_torque_w[ei, tid_body] = z
        global_force_at_com_w[ei, tid_body] = z
        local_force_b[ei, tid_body] = z
        local_torque_b[ei, tid_body] = z
        out_force_b[ei, tid_body] = z
        out_torque_b[ei, tid_body] = z

    return reset


@wp.kernel
def reset_wrench_composer_mask(
    env_mask: wp.array(dtype=wp.bool),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Zero all 7 wrench composer buffers for environments matching the mask.

    Dispatched with ``dim=(num_envs, num_bodies)``.
    """
    tid_env, tid_body = wp.tid()
    if env_mask[tid_env]:
        z = wp.vec3f(0.0)
        global_force_w[tid_env, tid_body] = z
        global_torque_w[tid_env, tid_body] = z
        global_force_at_com_w[tid_env, tid_body] = z
        local_force_b[tid_env, tid_body] = z
        local_torque_b[tid_env, tid_body] = z
        out_force_b[tid_env, tid_body] = z
        out_torque_b[tid_env, tid_body] = z
