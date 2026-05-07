# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-only camera orientation convention conversions.

Mirrors :func:`isaaclab.utils.math.convert_camera_frame_orientation_convention` for
``wp.array(dtype=wp.quatf)`` inputs without involving torch. Each conversion is a
fixed quaternion post-multiplication, with the convention quaternions baked in at
import time from a small numpy computation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

# --- Conversion quaternion derivation ---------------------------------------------------------
#
# The torch reference (``isaaclab.utils.math.convert_camera_frame_orientation_convention``)
# expresses each conversion as a 3x3 rotation matrix multiplied on the right of the source
# rotation matrix. Equivalently, in quaternion form ``q_target = q_src * q_conv``.
#
# Conventions (each describing the camera local axes in their frame):
#   * opengl: forward = -Z, up = +Y
#   * ros:    forward = +Z, up = -Y     (180 deg rotation around X relative to opengl)
#   * world:  forward = +X, up = +Z
#
# All quaternions are stored in (x, y, z, w) order to match :class:`warp.quatf`.


def _euler_xyz_intrinsic_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Intrinsic XYZ Euler angles to a 3x3 rotation matrix (R = Rx @ Ry @ Rz)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rym = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rzm = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rxm @ rym @ rzm


def _matrix_to_quat_xyzw(rotm: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a quaternion ``(x, y, z, w)``.

    Mirrors :func:`isaaclab.utils.math.quat_from_matrix` (the pytorch3d
    algorithm) so the sign convention matches the torch reference for any
    given input.
    """
    m00, m01, m02 = float(rotm[0, 0]), float(rotm[0, 1]), float(rotm[0, 2])
    m10, m11, m12 = float(rotm[1, 0]), float(rotm[1, 1]), float(rotm[1, 2])
    m20, m21, m22 = float(rotm[2, 0]), float(rotm[2, 1]), float(rotm[2, 2])

    q_abs_sq = (
        1.0 + m00 + m11 + m22,  # 4 * w^2
        1.0 + m00 - m11 - m22,  # 4 * x^2
        1.0 - m00 + m11 - m22,  # 4 * y^2
        1.0 - m00 - m11 + m22,  # 4 * z^2
    )
    q_abs = tuple(math.sqrt(max(0.0, v)) for v in q_abs_sq)

    # Candidate (x, y, z, w) tuples — one per branch.
    candidates = (
        (m21 - m12, m02 - m20, m10 - m01, q_abs[0] ** 2),
        (q_abs[1] ** 2, m10 + m01, m02 + m20, m21 - m12),
        (m10 + m01, q_abs[2] ** 2, m12 + m21, m02 - m20),
        (m20 + m02, m21 + m12, q_abs[3] ** 2, m10 - m01),
    )

    # Pick the branch with the largest q_abs (best-conditioned). Match the
    # torch reference's denominator floor of 0.1 to keep the algorithm bit-equivalent.
    idx = max(range(4), key=lambda i: q_abs[i])
    cand = candidates[idx]
    denom = 2.0 * max(q_abs[idx], 0.1)
    return cand[0] / denom, cand[1] / denom, cand[2] / denom, cand[3] / denom


# 180 deg rotation around X axis: q = (1, 0, 0, 0) in (x, y, z, w). 180-deg rotations are
# self-inverse, so the same quaternion serves both directions for ros<->opengl.
_Q_FLIP_YZ = (1.0, 0.0, 0.0, 0.0)
# The torch reference (``isaaclab.utils.math.convert_camera_frame_orientation_convention``)
# post-multiplies by matrix_from_euler([pi/2, -pi/2, 0], "XYZ") for world->opengl and by
# the *transpose* of the same matrix for opengl->world. As a quaternion, the transpose of
# a unit rotation matrix is the inverse — i.e. the conjugate of the corresponding quat.
_Q_WORLD_TO_OPENGL = _matrix_to_quat_xyzw(_euler_xyz_intrinsic_to_matrix(math.pi / 2, -math.pi / 2, 0.0))
_Q_OPENGL_TO_WORLD = (
    -_Q_WORLD_TO_OPENGL[0],
    -_Q_WORLD_TO_OPENGL[1],
    -_Q_WORLD_TO_OPENGL[2],
    _Q_WORLD_TO_OPENGL[3],
)


def _identity() -> tuple[float, float, float, float]:
    return 0.0, 0.0, 0.0, 1.0


def _quat_mul_xyzw(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product on (x, y, z, w) quaternions, returning a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _resolve_conversion_quat(
    origin: Literal["opengl", "ros", "world"],
    target: Literal["opengl", "ros", "world"],
) -> tuple[float, float, float, float]:
    """Return the right-multiplication quaternion that takes ``origin`` to ``target``."""
    if origin == target:
        return _identity()

    # First take origin -> opengl on the right.
    if origin == "ros":
        to_opengl = _Q_FLIP_YZ
    elif origin == "world":
        to_opengl = _Q_WORLD_TO_OPENGL
    else:  # origin == "opengl"
        to_opengl = _identity()

    # Then take opengl -> target on the right.
    if target == "ros":
        from_opengl = _Q_FLIP_YZ
    elif target == "world":
        from_opengl = _Q_OPENGL_TO_WORLD
    else:  # target == "opengl"
        from_opengl = _identity()

    # Combined right-multiplier: q_target = q_src * to_opengl * from_opengl
    return _quat_mul_xyzw(to_opengl, from_opengl)


@wp.kernel
def _quat_post_multiply_kernel(
    src: wp.array(dtype=wp.quatf),
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    dst: wp.array(dtype=wp.quatf),
):
    """For each i, dst[i] = src[i] * (qx, qy, qz, qw)."""
    tid = wp.tid()
    s = src[tid]
    sx = s[0]
    sy = s[1]
    sz = s[2]
    sw = s[3]
    # Hamilton product (x, y, z, w) order.
    rx = sw * qx + sx * qw + sy * qz - sz * qy
    ry = sw * qy - sx * qz + sy * qw + sz * qx
    rz = sw * qz + sx * qy - sy * qx + sz * qw
    rw = sw * qw - sx * qx - sy * qy - sz * qz
    dst[tid] = wp.quatf(rx, ry, rz, rw)


@wp.kernel
def _quat_xyzw_post_multiply_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    """Variant for ``(N, 4)`` float arrays storing (x, y, z, w) per row."""
    tid = wp.tid()
    sx = src[tid, 0]
    sy = src[tid, 1]
    sz = src[tid, 2]
    sw = src[tid, 3]
    dst[tid, 0] = sw * qx + sx * qw + sy * qz - sz * qy
    dst[tid, 1] = sw * qy - sx * qz + sy * qw + sz * qx
    dst[tid, 2] = sw * qz + sx * qy - sy * qx + sz * qw
    dst[tid, 3] = sw * qw - sx * qx - sy * qy - sz * qz


def convert_quat_array(
    src: wp.array,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> wp.array:
    """Convert a quaternion array from ``origin`` to ``target`` camera convention.

    Args:
        src: Source quaternions. Either ``wp.array(dtype=wp.quatf, shape=(N,))`` or
            ``wp.array(dtype=wp.float32, shape=(N, 4))`` with ``(x, y, z, w)`` rows.
        origin: Source convention.
        target: Target convention.

    Returns:
        A freshly-allocated :class:`warp.array` of the same dtype/shape with the
        converted quaternions.
    """
    qx, qy, qz, qw = _resolve_conversion_quat(origin, target)
    if origin == target:
        # Defensive copy so callers can mutate the result without touching ``src``.
        out = wp.empty_like(src)
        wp.copy(out, src)
        return out

    if src.dtype is wp.quatf:
        n = int(src.shape[0])
        dst = wp.empty(n, dtype=wp.quatf, device=src.device)
        wp.launch(
            _quat_post_multiply_kernel,
            dim=n,
            inputs=[src, float(qx), float(qy), float(qz), float(qw)],
            outputs=[dst],
            device=src.device,
        )
        return dst

    if src.dtype is wp.float32 and len(src.shape) == 2 and src.shape[1] == 4:
        n = int(src.shape[0])
        dst = wp.empty((n, 4), dtype=wp.float32, device=src.device)
        wp.launch(
            _quat_xyzw_post_multiply_kernel,
            dim=n,
            inputs=[src, float(qx), float(qy), float(qz), float(qw)],
            outputs=[dst],
            device=src.device,
        )
        return dst

    raise TypeError(
        f"convert_quat_array: unsupported source array (dtype={src.dtype}, shape={src.shape})."
        " Expected wp.quatf shape (N,) or wp.float32 shape (N, 4)."
    )
