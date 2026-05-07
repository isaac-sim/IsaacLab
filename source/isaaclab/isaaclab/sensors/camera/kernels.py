# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels backing the standard :class:`~isaaclab.sensors.camera.Camera`.

Each kernel keeps the camera's runtime fully Warp-native: there is no torch
involvement in the per-step pose / intrinsic / frame-counter updates.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def masked_increment_int64_kernel(
    mask: wp.array(dtype=wp.bool),
    counter: wp.array(dtype=wp.int64),
):
    """Increment ``counter[i]`` by one for every ``i`` where ``mask[i]`` is true."""
    tid = wp.tid()
    if mask[tid]:
        counter[tid] = counter[tid] + wp.int64(1)


@wp.kernel
def masked_set_int64_kernel(
    mask: wp.array(dtype=wp.bool),
    value: wp.int64,
    target: wp.array(dtype=wp.int64),
):
    """Overwrite ``target[i]`` with ``value`` for every ``i`` where ``mask[i]`` is true."""
    tid = wp.tid()
    if mask[tid]:
        target[tid] = value


@wp.kernel
def masked_set_vec3f_kernel(
    mask: wp.array(dtype=wp.bool),
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
):
    """For every ``i`` where ``mask[i]`` is true, assign ``dst[i] = src[i]``."""
    tid = wp.tid()
    if mask[tid]:
        dst[tid] = src[tid]


@wp.kernel
def masked_set_quatf_kernel(
    mask: wp.array(dtype=wp.bool),
    src: wp.array(dtype=wp.quatf),
    dst: wp.array(dtype=wp.quatf),
):
    """For every ``i`` where ``mask[i]`` is true, assign ``dst[i] = src[i]``."""
    tid = wp.tid()
    if mask[tid]:
        dst[tid] = src[tid]


@wp.kernel
def write_intrinsic_matrices_kernel(
    env_ids: wp.array(dtype=wp.int32),
    fx: wp.array(dtype=wp.float32),
    fy: wp.array(dtype=wp.float32),
    cx: wp.array(dtype=wp.float32),
    cy: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32, ndim=3),
):
    """Scatter intrinsic-matrix scalars (``f_x``, ``f_y``, ``c_x``, ``c_y``, 1.0)
    into ``target[env_ids[t], :, :]`` for ``t`` in ``[0, env_ids.shape[0])``.
    """
    tid = wp.tid()
    i = env_ids[tid]
    target[i, 0, 0] = fx[tid]
    target[i, 0, 2] = cx[tid]
    target[i, 1, 1] = fy[tid]
    target[i, 1, 2] = cy[tid]
    target[i, 2, 2] = wp.float32(1.0)


@wp.kernel
def vec3f_view_to_torch_layout_kernel(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    """Copy from a ``(N,)`` array of ``wp.vec3f`` into a ``(N, 3)`` float32 array."""
    tid = wp.tid()
    v = src[tid]
    dst[tid, 0] = v[0]
    dst[tid, 1] = v[1]
    dst[tid, 2] = v[2]


@wp.kernel
def quatf_view_to_torch_layout_kernel(
    src: wp.array(dtype=wp.quatf),
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    """Copy from a ``(N,)`` array of ``wp.quatf`` into a ``(N, 4)`` float32 array."""
    tid = wp.tid()
    q = src[tid]
    dst[tid, 0] = q[0]
    dst[tid, 1] = q[1]
    dst[tid, 2] = q[2]
    dst[tid, 3] = q[3]


@wp.kernel
def torch_layout_to_vec3f_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    dst: wp.array(dtype=wp.vec3f),
):
    """Copy from a ``(N, 3)`` float32 array into a ``(N,)`` ``wp.vec3f`` array."""
    tid = wp.tid()
    dst[tid] = wp.vec3f(src[tid, 0], src[tid, 1], src[tid, 2])


@wp.kernel
def torch_layout_to_quatf_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    dst: wp.array(dtype=wp.quatf),
):
    """Copy from a ``(N, 4)`` float32 array into a ``(N,)`` ``wp.quatf`` array."""
    tid = wp.tid()
    dst[tid] = wp.quatf(src[tid, 0], src[tid, 1], src[tid, 2], src[tid, 3])


@wp.kernel
def mask_to_indices_kernel(
    mask: wp.array(dtype=wp.bool),
    indices: wp.array(dtype=wp.int32),
    counter: wp.array(dtype=wp.int32),
):
    """Compact the indices where ``mask[i]`` is true into ``indices[0..counter[0])``.

    ``counter`` must be zero-initialized before launch. ``indices`` must have
    capacity at least ``mask.shape[0]``.
    """
    tid = wp.tid()
    if mask[tid]:
        idx = wp.atomic_add(counter, 0, 1)
        indices[idx] = wp.int32(tid)


@wp.kernel
def indices_to_mask_kernel(
    indices: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.bool),
):
    """For each ``i`` in ``indices``, set ``mask[indices[i]] = True``.

    ``mask`` must be zero-initialized before launch.
    """
    tid = wp.tid()
    mask[indices[tid]] = True


@wp.kernel
def scatter_vec3f_kernel(
    indices: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
):
    """``dst[indices[t]] = src[t]`` for ``t`` in ``[0, indices.shape[0])``."""
    tid = wp.tid()
    dst[indices[tid]] = src[tid]


@wp.kernel
def scatter_quatf_kernel(
    indices: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.quatf),
    dst: wp.array(dtype=wp.quatf),
):
    """``dst[indices[t]] = src[t]`` for ``t`` in ``[0, indices.shape[0])``."""
    tid = wp.tid()
    dst[indices[tid]] = src[tid]


@wp.kernel
def combine_frame_transforms_kernel(
    p1: wp.array(dtype=wp.vec3),
    q1: wp.array(dtype=wp.quatf),
    p2: wp.array(dtype=wp.vec3),
    q2: wp.array(dtype=wp.quatf),
    out_p: wp.array(dtype=wp.vec3),
    out_q: wp.array(dtype=wp.quatf),
):
    """Compose two rigid transforms: ``out = T1 ∘ T2`` (parent ∘ child).

    Mirrors :func:`isaaclab.utils.math.combine_frame_transforms`.
    """
    tid = wp.tid()
    out_p[tid] = p1[tid] + wp.quat_rotate(q1[tid], p2[tid])
    out_q[tid] = q1[tid] * q2[tid]


@wp.kernel
def gather_vec3f_kernel(
    indices: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
):
    """``dst[t] = src[indices[t]]`` for ``t`` in ``[0, indices.shape[0])``."""
    tid = wp.tid()
    dst[tid] = src[indices[tid]]


@wp.kernel
def gather_quatf_kernel(
    indices: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.quatf),
    dst: wp.array(dtype=wp.quatf),
):
    """``dst[t] = src[indices[t]]`` for ``t`` in ``[0, indices.shape[0])``."""
    tid = wp.tid()
    dst[tid] = src[indices[tid]]


@wp.kernel
def quat_inverse_apply_kernel(
    q: wp.array(dtype=wp.quatf),
    v: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    """``out[i] = quat_rotate(quat_inverse(q[i]), v[i])`` — rotate ``v`` by the inverse of ``q``."""
    tid = wp.tid()
    out[tid] = wp.quat_rotate_inv(q[tid], v[tid])


@wp.kernel
def quat_inverse_multiply_kernel(
    q1: wp.array(dtype=wp.quatf),
    q2: wp.array(dtype=wp.quatf),
    out: wp.array(dtype=wp.quatf),
):
    """``out[i] = quat_inverse(q1[i]) * q2[i]``."""
    tid = wp.tid()
    out[tid] = wp.quat_inverse(q1[tid]) * q2[tid]


@wp.kernel
def vec3_subtract_kernel(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    """``out[i] = a[i] - b[i]``."""
    tid = wp.tid()
    out[tid] = a[tid] - b[tid]


@wp.kernel
def look_at_quat_kernel(
    eyes: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    up_world: wp.vec3,
    out: wp.array(dtype=wp.quatf),
):
    """For each camera ``i``, compute the OpenGL-convention look-at quaternion that
    points the camera from ``eyes[i]`` toward ``targets[i]`` with the supplied
    world up-axis. The result is stored in ``out[i]`` as ``wp.quatf`` (x, y, z, w).

    Mirrors :func:`isaaclab.utils.math.create_rotation_matrix_from_view` followed by
    :func:`isaaclab.utils.math.quat_from_matrix` so external behavior is preserved
    while the implementation remains pure warp.
    """
    tid = wp.tid()
    # OpenGL camera convention: forward = -Z, up = +Y, right = +X.
    forward = wp.normalize(targets[tid] - eyes[tid])
    right = wp.normalize(wp.cross(forward, up_world))
    cam_up = wp.cross(right, forward)
    # Rotation matrix columns: (right, up, -forward)
    m00 = right[0]
    m01 = cam_up[0]
    m02 = -forward[0]
    m10 = right[1]
    m11 = cam_up[1]
    m12 = -forward[1]
    m20 = right[2]
    m21 = cam_up[2]
    m22 = -forward[2]
    # Convert to quaternion (xyzw) using the pytorch3d-style algorithm.
    # 4 candidate q_abs^2 components: (4w^2, 4x^2, 4y^2, 4z^2).
    qaw = wp.sqrt(wp.max(0.0, 1.0 + m00 + m11 + m22))
    qax = wp.sqrt(wp.max(0.0, 1.0 + m00 - m11 - m22))
    qay = wp.sqrt(wp.max(0.0, 1.0 - m00 + m11 - m22))
    qaz = wp.sqrt(wp.max(0.0, 1.0 - m00 - m11 + m22))
    # Pick the largest one and recover signed components from off-diagonal entries.
    if qaw >= qax and qaw >= qay and qaw >= qaz:
        denom = 2.0 * wp.max(qaw, 0.1)
        qx = (m21 - m12) / denom
        qy = (m02 - m20) / denom
        qz = (m10 - m01) / denom
        qw = qaw * qaw / denom
    elif qax >= qay and qax >= qaz:
        denom = 2.0 * wp.max(qax, 0.1)
        qx = qax * qax / denom
        qy = (m10 + m01) / denom
        qz = (m02 + m20) / denom
        qw = (m21 - m12) / denom
    elif qay >= qaz:
        denom = 2.0 * wp.max(qay, 0.1)
        qx = (m10 + m01) / denom
        qy = qay * qay / denom
        qz = (m12 + m21) / denom
        qw = (m02 - m20) / denom
    else:
        denom = 2.0 * wp.max(qaz, 0.1)
        qx = (m20 + m02) / denom
        qy = (m21 + m12) / denom
        qz = qaz * qaz / denom
        qw = (m10 - m01) / denom
    out[tid] = wp.quatf(qx, qy, qz, qw)
