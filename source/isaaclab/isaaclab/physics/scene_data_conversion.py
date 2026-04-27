# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion dispatcher for transform format conversions.

Selects and launches the appropriate Warp kernel for any
``(source_format, target_format)`` pair, including quaternion convention
swizzle and matrix layout transpose as post-processing steps.
"""

from __future__ import annotations

import warp as wp

from . import scene_data_kernels as K
from .scene_data_types import (
    Mat44Transforms,
    MatrixLayout,
    QuaternionConvention,
    TransformArrayData,
    TransformData,
    TransformFormat,
    Vec3Mat33Transforms,
    Vec3QuatTransforms,
)

_EMPTY_INDEX_MAP: dict[str, wp.array] = {}


def _get_empty_index_map(device: str) -> wp.array:
    """Return a cached zero-length int32 array on the given device."""
    if device not in _EMPTY_INDEX_MAP:
        _EMPTY_INDEX_MAP[device] = wp.empty(0, dtype=wp.int32, device=device)
    return _EMPTY_INDEX_MAP[device]


class ConversionDispatcher:
    """Dispatches transform format conversions via Warp kernels.

    All conversions run entirely on GPU. The dispatcher selects the
    appropriate kernel based on ``(source.format, target.format)`` and
    optionally applies quaternion convention swizzle or matrix layout
    transpose as post-processing steps.
    """

    @staticmethod
    def can_passthrough(
        source: TransformData,
        target_format: TransformFormat,
        *,
        quat_convention: QuaternionConvention = QuaternionConvention.XYZW,
        matrix_layout: MatrixLayout = MatrixLayout.ROW_MAJOR,
        double_precision: bool = False,
    ) -> bool:
        """Check whether source data can be returned directly without conversion.

        Args:
            source: Source transform data.
            target_format: Desired output format.
            quat_convention: Desired quaternion convention (for VEC3_QUAT).
            matrix_layout: Desired matrix layout (for VEC3_MAT33 / MAT44).
            double_precision: Whether MAT44 output should use float64.

        Returns:
            ``True`` if no conversion or copy is needed.
        """
        if source.format != target_format:
            return False
        if target_format == TransformFormat.VEC3_QUAT:
            assert isinstance(source, Vec3QuatTransforms)
            return source.quat_convention == quat_convention
        if target_format == TransformFormat.VEC3_MAT33:
            assert isinstance(source, Vec3Mat33Transforms)
            return source.layout == matrix_layout
        if target_format == TransformFormat.MAT44:
            assert isinstance(source, Mat44Transforms)
            return source.layout == matrix_layout and source.double_precision == double_precision
        return True

    @staticmethod
    def convert(  # noqa: C901
        source: TransformData,
        target: TransformData,
        *,
        stream: wp.Stream | None = None,
        index_map: wp.array | None = None,
    ) -> None:
        """Convert source transform data into a pre-allocated target buffer.

        All work is done on GPU via Warp kernels. When *stream* is provided,
        kernels are launched on that stream for deferred execution.

        Args:
            source: Source transform data (read-only).
            target: Pre-allocated target buffer to write into.
            stream: Optional CUDA stream for deferred execution.
            index_map: Optional index remapping array. When provided,
                output element *i* is written to ``target[index_map[i]]``.
        """
        device = source.device
        imap = index_map if index_map is not None else _get_empty_index_map(device)
        imap_len = int(index_map.shape[0]) if index_map is not None else 0
        dim = imap_len if index_map is not None else source.count

        def _launch(kernel, inputs):
            if stream is not None:
                with wp.ScopedStream(stream):
                    wp.launch(kernel, dim=dim, inputs=inputs, device=device)
            else:
                wp.launch(kernel, dim=dim, inputs=inputs, device=device)

        src_fmt = source.format
        dst_fmt = target.format

        # --- vec3_quat source ---
        if src_fmt == TransformFormat.VEC3_QUAT:
            assert isinstance(source, Vec3QuatTransforms)
            sp, so = source.positions, source.orientations
            if dst_fmt == TransformFormat.VEC3_QUAT:
                assert isinstance(target, Vec3QuatTransforms)
                _launch(
                    K.vec3_quat_to_vec3_quat_kernel, [sp, so, target.positions, target.orientations, imap, imap_len]
                )
            elif dst_fmt == TransformFormat.VEC3_MAT33:
                assert isinstance(target, Vec3Mat33Transforms)
                _launch(K.vec3_quat_to_vec3_mat33_kernel, [sp, so, target.positions, target.rotations, imap, imap_len])
            elif dst_fmt == TransformFormat.TRANSFORM:
                assert isinstance(target, TransformArrayData)
                _launch(K.vec3_quat_to_transform_kernel, [sp, so, target.transforms, imap, imap_len])
            elif dst_fmt == TransformFormat.MAT44:
                assert isinstance(target, Mat44Transforms)
                if target.double_precision:
                    _launch(K.vec3_quat_to_mat44d_kernel, [sp, so, target.matrices, imap, imap_len])
                else:
                    _launch(K.vec3_quat_to_mat44f_kernel, [sp, so, target.matrices, imap, imap_len])

        # --- vec3_mat33 source ---
        elif src_fmt == TransformFormat.VEC3_MAT33:
            assert isinstance(source, Vec3Mat33Transforms)
            sp, sr = source.positions, source.rotations
            if dst_fmt == TransformFormat.VEC3_QUAT:
                assert isinstance(target, Vec3QuatTransforms)
                _launch(
                    K.vec3_mat33_to_vec3_quat_kernel, [sp, sr, target.positions, target.orientations, imap, imap_len]
                )
            elif dst_fmt == TransformFormat.VEC3_MAT33:
                assert isinstance(target, Vec3Mat33Transforms)
                _launch(K.vec3_mat33_to_vec3_mat33_kernel, [sp, sr, target.positions, target.rotations, imap, imap_len])
            elif dst_fmt == TransformFormat.TRANSFORM:
                assert isinstance(target, TransformArrayData)
                _launch(K.vec3_mat33_to_transform_kernel, [sp, sr, target.transforms, imap, imap_len])
            elif dst_fmt == TransformFormat.MAT44:
                assert isinstance(target, Mat44Transforms)
                _launch(K.vec3_mat33_to_mat44f_kernel, [sp, sr, target.matrices, imap, imap_len])

        # --- transform source ---
        elif src_fmt == TransformFormat.TRANSFORM:
            assert isinstance(source, TransformArrayData)
            st = source.transforms
            if dst_fmt == TransformFormat.VEC3_QUAT:
                assert isinstance(target, Vec3QuatTransforms)
                _launch(K.transform_to_vec3_quat_kernel, [st, target.positions, target.orientations, imap, imap_len])
            elif dst_fmt == TransformFormat.VEC3_MAT33:
                assert isinstance(target, Vec3Mat33Transforms)
                _launch(K.transform_to_vec3_mat33_kernel, [st, target.positions, target.rotations, imap, imap_len])
            elif dst_fmt == TransformFormat.TRANSFORM:
                assert isinstance(target, TransformArrayData)
                _launch(K.transform_to_transform_kernel, [st, target.transforms, imap, imap_len])
            elif dst_fmt == TransformFormat.MAT44:
                assert isinstance(target, Mat44Transforms)
                if target.double_precision:
                    _launch(K.transform_to_mat44d_kernel, [st, target.matrices, imap, imap_len])
                else:
                    _launch(K.transform_to_mat44f_kernel, [st, target.matrices, imap, imap_len])

        # --- mat44 source ---
        elif src_fmt == TransformFormat.MAT44:
            assert isinstance(source, Mat44Transforms)
            sm = source.matrices
            if dst_fmt == TransformFormat.VEC3_QUAT:
                assert isinstance(target, Vec3QuatTransforms)
                if source.double_precision:
                    _launch(K.mat44d_to_vec3_quat_kernel, [sm, target.positions, target.orientations, imap, imap_len])
                else:
                    _launch(K.mat44f_to_vec3_quat_kernel, [sm, target.positions, target.orientations, imap, imap_len])
            elif dst_fmt == TransformFormat.VEC3_MAT33:
                assert isinstance(target, Vec3Mat33Transforms)
                if source.double_precision:
                    raise NotImplementedError("mat44d -> vec3_mat33 not yet supported")
                _launch(K.mat44f_to_vec3_mat33_kernel, [sm, target.positions, target.rotations, imap, imap_len])
            elif dst_fmt == TransformFormat.TRANSFORM:
                assert isinstance(target, TransformArrayData)
                if source.double_precision:
                    _launch(K.mat44d_to_transform_kernel, [sm, target.transforms, imap, imap_len])
                else:
                    _launch(K.mat44f_to_transform_kernel, [sm, target.transforms, imap, imap_len])
            elif dst_fmt == TransformFormat.MAT44:
                assert isinstance(target, Mat44Transforms)
                _launch(K.mat44f_to_mat44f_kernel, [sm, target.matrices, imap, imap_len])

        # --- Post-processing: quaternion convention swizzle ---
        if dst_fmt == TransformFormat.VEC3_QUAT:
            assert isinstance(target, Vec3QuatTransforms)
            src_conv = QuaternionConvention.XYZW
            if src_fmt == TransformFormat.VEC3_QUAT:
                assert isinstance(source, Vec3QuatTransforms)
                src_conv = source.quat_convention
            if src_conv != target.quat_convention:
                empty_imap = _get_empty_index_map(device)
                if target.quat_convention == QuaternionConvention.WXYZ:
                    _launch(K.quat_xyzw_to_wxyz_kernel, [target.orientations, target.orientations, empty_imap, 0])
                else:
                    _launch(K.quat_wxyz_to_xyzw_kernel, [target.orientations, target.orientations, empty_imap, 0])

        # --- Post-processing: matrix layout transpose ---
        if dst_fmt == TransformFormat.MAT44:
            assert isinstance(target, Mat44Transforms)
            if target.layout == MatrixLayout.COLUMN_MAJOR:
                empty_imap = _get_empty_index_map(device)
                if target.double_precision:
                    _launch(K.mat44d_transpose_kernel, [target.matrices, target.matrices, empty_imap, 0])
                else:
                    _launch(K.mat44f_transpose_kernel, [target.matrices, target.matrices, empty_imap, 0])

        if dst_fmt == TransformFormat.VEC3_MAT33:
            assert isinstance(target, Vec3Mat33Transforms)
            if target.layout == MatrixLayout.COLUMN_MAJOR:
                empty_imap = _get_empty_index_map(device)
                _launch(K.mat33f_transpose_kernel, [target.rotations, target.rotations, empty_imap, 0])
