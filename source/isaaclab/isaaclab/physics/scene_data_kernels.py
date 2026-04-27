# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp GPU kernels for transform format conversion.

Provides a 4x4 grid of conversion kernels between the four transform
representations used in Isaac Lab, plus quaternion convention swizzle
and matrix layout transpose kernels.

All kernels support an optional ``index_map`` for subset scatter writes:
when provided, output element *i* is written to ``output[index_map[i]]``
instead of ``output[i]``.  Pass ``None`` (empty array with shape 0) to
disable index mapping.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@wp.func
def _output_idx(tid: int, index_map: wp.array(dtype=wp.int32), index_map_len: int) -> int:  # type: ignore
    """Return mapped output index, or *tid* when index_map is empty."""
    if index_map_len > 0:
        return index_map[tid]
    return tid


@wp.func
def _quat_to_mat33(q: wp.quatf) -> wp.mat33f:  # type: ignore
    """Convert a quaternion (XYZW) to a 3x3 rotation matrix (row-major)."""
    return wp.quat_to_matrix(q)


@wp.func
def _mat33_to_quat(m: wp.mat33f) -> wp.quatf:  # type: ignore
    """Convert a 3x3 rotation matrix (row-major) to a quaternion (XYZW)."""
    return wp.quat_from_matrix(m)


@wp.func
def _transform_to_mat44f(t: wp.transformf) -> wp.mat44f:  # type: ignore
    """Convert a packed transform to a 4x4 matrix (row-major, float32)."""
    m44 = wp.math.transform_to_matrix(t)
    return wp.mat44f(  # type: ignore
        m44[0, 0],
        m44[0, 1],
        m44[0, 2],
        m44[0, 3],
        m44[1, 0],
        m44[1, 1],
        m44[1, 2],
        m44[1, 3],
        m44[2, 0],
        m44[2, 1],
        m44[2, 2],
        m44[2, 3],
        m44[3, 0],
        m44[3, 1],
        m44[3, 2],
        m44[3, 3],
    )


@wp.func
def _mat44f_to_transform(m: wp.mat44f) -> wp.transformf:  # type: ignore
    """Convert a 4x4 matrix (row-major, float32) to a packed transform."""
    pos = wp.vec3(m[0, 3], m[1, 3], m[2, 3])
    rot = wp.mat33f(  # type: ignore
        m[0, 0],
        m[0, 1],
        m[0, 2],
        m[1, 0],
        m[1, 1],
        m[1, 2],
        m[2, 0],
        m[2, 1],
        m[2, 2],
    )
    q = wp.quat_from_matrix(rot)
    return wp.transformf(pos, q)


# ---------------------------------------------------------------------------
# vec3_quat -> other formats
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def vec3_quat_to_vec3_quat_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_ori: wp.array(dtype=wp.quatf),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_ori: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Copy vec3+quat to vec3+quat (pass-through with optional index remap)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_pos[oid] = src_pos[tid]
    dst_ori[oid] = src_ori[tid]


@wp.kernel(enable_backward=False)
def vec3_quat_to_vec3_mat33_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_ori: wp.array(dtype=wp.quatf),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+quat to vec3+mat33."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_pos[oid] = src_pos[tid]
    dst_rot[oid] = _quat_to_mat33(src_ori[tid])


@wp.kernel(enable_backward=False)
def vec3_quat_to_transform_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_ori: wp.array(dtype=wp.quatf),  # type: ignore
    dst_tf: wp.array(dtype=wp.transformf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+quat to packed transformf."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_tf[oid] = wp.transformf(src_pos[tid], src_ori[tid])


@wp.kernel(enable_backward=False)
def vec3_quat_to_mat44f_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_ori: wp.array(dtype=wp.quatf),  # type: ignore
    dst_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+quat to 4x4 float32 matrix (row-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    t = wp.transformf(src_pos[tid], src_ori[tid])
    dst_mat[oid] = _transform_to_mat44f(t)


@wp.kernel(enable_backward=False)
def vec3_quat_to_mat44d_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_ori: wp.array(dtype=wp.quatf),  # type: ignore
    dst_mat: wp.array(dtype=wp.mat44d),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+quat to 4x4 float64 matrix (row-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    t = wp.transformf(src_pos[tid], src_ori[tid])
    m = wp.math.transform_to_matrix(t)
    dst_mat[oid] = wp.mat44d(  # type: ignore
        wp.float64(m[0, 0]),
        wp.float64(m[0, 1]),
        wp.float64(m[0, 2]),
        wp.float64(m[0, 3]),
        wp.float64(m[1, 0]),
        wp.float64(m[1, 1]),
        wp.float64(m[1, 2]),
        wp.float64(m[1, 3]),
        wp.float64(m[2, 0]),
        wp.float64(m[2, 1]),
        wp.float64(m[2, 2]),
        wp.float64(m[2, 3]),
        wp.float64(m[3, 0]),
        wp.float64(m[3, 1]),
        wp.float64(m[3, 2]),
        wp.float64(m[3, 3]),
    )


# ---------------------------------------------------------------------------
# vec3_mat33 -> other formats
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def vec3_mat33_to_vec3_quat_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_ori: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+mat33 to vec3+quat."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_pos[oid] = src_pos[tid]
    dst_ori[oid] = _mat33_to_quat(src_rot[tid])


@wp.kernel(enable_backward=False)
def vec3_mat33_to_vec3_mat33_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Copy vec3+mat33 to vec3+mat33 (pass-through with optional index remap)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_pos[oid] = src_pos[tid]
    dst_rot[oid] = src_rot[tid]


@wp.kernel(enable_backward=False)
def vec3_mat33_to_transform_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    dst_tf: wp.array(dtype=wp.transformf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+mat33 to packed transformf."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    q = _mat33_to_quat(src_rot[tid])
    dst_tf[oid] = wp.transformf(src_pos[tid], q)


@wp.kernel(enable_backward=False)
def vec3_mat33_to_mat44f_kernel(
    src_pos: wp.array(dtype=wp.vec3),  # type: ignore
    src_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    dst_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert vec3+mat33 to 4x4 float32 matrix (row-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    p = src_pos[tid]
    r = src_rot[tid]
    dst_mat[oid] = wp.mat44f(  # type: ignore
        r[0, 0],
        r[0, 1],
        r[0, 2],
        p[0],
        r[1, 0],
        r[1, 1],
        r[1, 2],
        p[1],
        r[2, 0],
        r[2, 1],
        r[2, 2],
        p[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------------
# transform -> other formats
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def transform_to_vec3_quat_kernel(
    src_tf: wp.array(dtype=wp.transformf),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_ori: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert packed transformf to vec3+quat."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_pos[oid] = wp.transform_get_translation(src_tf[tid])
    dst_ori[oid] = wp.transform_get_rotation(src_tf[tid])


@wp.kernel(enable_backward=False)
def transform_to_vec3_mat33_kernel(
    src_tf: wp.array(dtype=wp.transformf),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert packed transformf to vec3+mat33."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_pos[oid] = wp.transform_get_translation(src_tf[tid])
    dst_rot[oid] = _quat_to_mat33(wp.transform_get_rotation(src_tf[tid]))


@wp.kernel(enable_backward=False)
def transform_to_transform_kernel(
    src_tf: wp.array(dtype=wp.transformf),  # type: ignore
    dst_tf: wp.array(dtype=wp.transformf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Copy transformf to transformf (pass-through with optional index remap)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_tf[oid] = src_tf[tid]


@wp.kernel(enable_backward=False)
def transform_to_mat44f_kernel(
    src_tf: wp.array(dtype=wp.transformf),  # type: ignore
    dst_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert packed transformf to 4x4 float32 matrix (row-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_mat[oid] = _transform_to_mat44f(src_tf[tid])


@wp.kernel(enable_backward=False)
def transform_to_mat44d_kernel(
    src_tf: wp.array(dtype=wp.transformf),  # type: ignore
    dst_mat: wp.array(dtype=wp.mat44d),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert packed transformf to 4x4 float64 matrix (row-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    m = wp.math.transform_to_matrix(src_tf[tid])
    dst_mat[oid] = wp.mat44d(  # type: ignore
        wp.float64(m[0, 0]),
        wp.float64(m[0, 1]),
        wp.float64(m[0, 2]),
        wp.float64(m[0, 3]),
        wp.float64(m[1, 0]),
        wp.float64(m[1, 1]),
        wp.float64(m[1, 2]),
        wp.float64(m[1, 3]),
        wp.float64(m[2, 0]),
        wp.float64(m[2, 1]),
        wp.float64(m[2, 2]),
        wp.float64(m[2, 3]),
        wp.float64(m[3, 0]),
        wp.float64(m[3, 1]),
        wp.float64(m[3, 2]),
        wp.float64(m[3, 3]),
    )


# ---------------------------------------------------------------------------
# mat44 -> other formats (float32 versions)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def mat44f_to_vec3_quat_kernel(
    src_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_ori: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert 4x4 float32 matrix (row-major) to vec3+quat."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    t = _mat44f_to_transform(src_mat[tid])
    dst_pos[oid] = wp.transform_get_translation(t)
    dst_ori[oid] = wp.transform_get_rotation(t)


@wp.kernel(enable_backward=False)
def mat44f_to_vec3_mat33_kernel(
    src_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_rot: wp.array(dtype=wp.mat33f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert 4x4 float32 matrix (row-major) to vec3+mat33."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    m = src_mat[tid]
    dst_pos[oid] = wp.vec3(m[0, 3], m[1, 3], m[2, 3])
    dst_rot[oid] = wp.mat33f(  # type: ignore
        m[0, 0],
        m[0, 1],
        m[0, 2],
        m[1, 0],
        m[1, 1],
        m[1, 2],
        m[2, 0],
        m[2, 1],
        m[2, 2],
    )


@wp.kernel(enable_backward=False)
def mat44f_to_transform_kernel(
    src_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    dst_tf: wp.array(dtype=wp.transformf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert 4x4 float32 matrix (row-major) to packed transformf."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_tf[oid] = _mat44f_to_transform(src_mat[tid])


@wp.kernel(enable_backward=False)
def mat44f_to_mat44f_kernel(
    src_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    dst_mat: wp.array(dtype=wp.mat44f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Copy mat44f to mat44f (pass-through with optional index remap)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst_mat[oid] = src_mat[tid]


# ---------------------------------------------------------------------------
# mat44 -> other formats (float64 versions)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def mat44d_to_vec3_quat_kernel(
    src_mat: wp.array(dtype=wp.mat44d),  # type: ignore
    dst_pos: wp.array(dtype=wp.vec3),  # type: ignore
    dst_ori: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert 4x4 float64 matrix (row-major) to vec3+quat."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    m = src_mat[tid]
    dst_pos[oid] = wp.vec3(float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
    rot = wp.mat33f(  # type: ignore
        float(m[0, 0]),
        float(m[0, 1]),
        float(m[0, 2]),
        float(m[1, 0]),
        float(m[1, 1]),
        float(m[1, 2]),
        float(m[2, 0]),
        float(m[2, 1]),
        float(m[2, 2]),
    )
    dst_ori[oid] = wp.quat_from_matrix(rot)


@wp.kernel(enable_backward=False)
def mat44d_to_transform_kernel(
    src_mat: wp.array(dtype=wp.mat44d),  # type: ignore
    dst_tf: wp.array(dtype=wp.transformf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Convert 4x4 float64 matrix (row-major) to packed transformf."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    m = src_mat[tid]
    pos = wp.vec3(float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
    rot = wp.mat33f(  # type: ignore
        float(m[0, 0]),
        float(m[0, 1]),
        float(m[0, 2]),
        float(m[1, 0]),
        float(m[1, 1]),
        float(m[1, 2]),
        float(m[2, 0]),
        float(m[2, 1]),
        float(m[2, 2]),
    )
    q = wp.quat_from_matrix(rot)
    dst_tf[oid] = wp.transformf(pos, q)


# ---------------------------------------------------------------------------
# Quaternion convention swizzle kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def quat_xyzw_to_wxyz_kernel(
    src: wp.array(dtype=wp.quatf),  # type: ignore
    dst: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Swizzle quaternion from XYZW to WXYZ storage order.

    Warp stores quatf as (x,y,z,w). This kernel reinterprets the output
    so the w component occupies the first slot when read as raw floats.
    """
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    q = src[tid]
    dst[oid] = wp.quatf(q[3], q[0], q[1], q[2])


@wp.kernel(enable_backward=False)
def quat_wxyz_to_xyzw_kernel(
    src: wp.array(dtype=wp.quatf),  # type: ignore
    dst: wp.array(dtype=wp.quatf),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Swizzle quaternion from WXYZ to XYZW storage order."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    q = src[tid]
    dst[oid] = wp.quatf(q[1], q[2], q[3], q[0])


# ---------------------------------------------------------------------------
# Matrix layout transpose kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def mat44f_transpose_kernel(
    src: wp.array(dtype=wp.mat44f),  # type: ignore
    dst: wp.array(dtype=wp.mat44f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Transpose 4x4 float32 matrices (row-major <-> column-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst[oid] = wp.transpose(src[tid])


@wp.kernel(enable_backward=False)
def mat44d_transpose_kernel(
    src: wp.array(dtype=wp.mat44d),  # type: ignore
    dst: wp.array(dtype=wp.mat44d),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Transpose 4x4 float64 matrices (row-major <-> column-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst[oid] = wp.transpose(src[tid])


@wp.kernel(enable_backward=False)
def mat33f_transpose_kernel(
    src: wp.array(dtype=wp.mat33f),  # type: ignore
    dst: wp.array(dtype=wp.mat33f),  # type: ignore
    index_map: wp.array(dtype=wp.int32),  # type: ignore
    index_map_len: int,
):
    """Transpose 3x3 float32 matrices (row-major <-> column-major)."""
    tid = wp.tid()
    oid = _output_idx(tid, index_map, index_map_len)
    dst[oid] = wp.transpose(src[tid])
