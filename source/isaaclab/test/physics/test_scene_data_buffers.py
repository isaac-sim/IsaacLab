# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TransformBufferPool."""

from __future__ import annotations

import numpy as np
import warp as wp

from isaaclab.physics.scene_data_buffers import TransformBufferPool
from isaaclab.physics.scene_data_types import (
    Mat44Transforms,
    MatrixLayout,
    QuaternionConvention,
    TransformArrayData,
    TransformFormat,
    Vec3QuatTransforms,
)

DEVICE = "cuda:0"


def _make_source_vec3_quat(n=4):
    """Create a Vec3QuatTransforms with test data."""
    positions = wp.from_numpy(np.array([[1.0, 2.0, 3.0]] * n, dtype=np.float32), dtype=wp.vec3, device=DEVICE)
    quats = wp.from_numpy(np.array([[0.0, 0.0, 0.0, 1.0]] * n, dtype=np.float32), dtype=wp.quatf, device=DEVICE)
    return Vec3QuatTransforms(
        count=n,
        device=DEVICE,
        positions=positions,
        orientations=quats,
        quat_convention=QuaternionConvention.XYZW,
    )


def _make_source_transform(n=4):
    """Create a TransformArrayData with test data."""
    tf_np = np.zeros((n, 7), dtype=np.float32)
    tf_np[:, :3] = [1.0, 2.0, 3.0]
    tf_np[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    transforms = wp.from_numpy(tf_np, dtype=wp.transformf, device=DEVICE)
    return TransformArrayData(
        count=n,
        device=DEVICE,
        transforms=transforms,
    )


class TestPassthrough:
    def test_format_match_returns_source(self):
        """When requested format matches source, return source directly."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        result = pool.get_or_convert(
            source,
            TransformFormat.VEC3_QUAT,
            generation=1,
            allow_passthrough=True,
        )
        assert result is source

    def test_transform_passthrough(self):
        """TransformArrayData -> TRANSFORM should passthrough."""
        pool = TransformBufferPool()
        source = _make_source_transform()

        result = pool.get_or_convert(
            source,
            TransformFormat.TRANSFORM,
            generation=1,
            allow_passthrough=True,
        )
        assert result is source

    def test_no_passthrough_when_disabled(self):
        """When allow_passthrough=False, result should be different object."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        result = pool.get_or_convert(
            source,
            TransformFormat.VEC3_QUAT,
            generation=1,
            allow_passthrough=False,
        )
        assert result is not source
        assert isinstance(result, Vec3QuatTransforms)

    def test_no_passthrough_when_convention_differs(self):
        """When quat convention differs, conversion should happen."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        result = pool.get_or_convert(
            source,
            TransformFormat.VEC3_QUAT,
            generation=1,
            quat_convention=QuaternionConvention.WXYZ,
            allow_passthrough=True,
        )
        assert result is not source

    def test_no_passthrough_with_index_map(self):
        """When index_map is provided, passthrough is disabled."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()
        index_map = wp.from_numpy(np.array([0, 1], dtype=np.int32), device=DEVICE)

        result = pool.get_or_convert(
            source,
            TransformFormat.VEC3_QUAT,
            generation=1,
            allow_passthrough=True,
            index_map=index_map,
        )
        assert result is not source


class TestGenerationCache:
    def test_same_generation_returns_cached(self):
        """Second call with same generation should return cached result."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        result1 = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=1)
        result2 = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=1)
        assert result1 is result2

    def test_different_generation_reconverts(self):
        """Different generation should trigger re-conversion."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        result1 = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=1)
        result2 = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=2)
        assert isinstance(result1, TransformArrayData)
        assert isinstance(result2, TransformArrayData)


class TestBufferReuse:
    def test_cross_frame_reuses_buffer(self):
        """New generation should reuse the same allocated buffer."""
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        result1 = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=1)
        assert isinstance(result1, TransformArrayData)
        ptr1 = result1.transforms.ptr

        result2 = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=2)
        assert isinstance(result2, TransformArrayData)
        ptr2 = result2.transforms.ptr

        assert ptr1 == ptr2

    def test_reallocates_when_count_changes(self):
        """When source count changes, buffer should be reallocated."""
        pool = TransformBufferPool()

        source_small = _make_source_vec3_quat(n=2)
        result1 = pool.get_or_convert(source_small, TransformFormat.TRANSFORM, generation=1)
        assert isinstance(result1, TransformArrayData)

        source_large = _make_source_vec3_quat(n=8)
        result2 = pool.get_or_convert(source_large, TransformFormat.TRANSFORM, generation=2)
        assert isinstance(result2, TransformArrayData)
        assert result2.count == 8


class TestFormatConversion:
    def test_vec3_quat_to_transform(self):
        pool = TransformBufferPool()
        source = _make_source_vec3_quat(n=2)

        result = pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=1)
        assert isinstance(result, TransformArrayData)
        assert result.count == 2
        wp.synchronize_device(DEVICE)

        tf = result.transforms.numpy()
        np.testing.assert_allclose(tf[0][:3], [1.0, 2.0, 3.0], atol=1e-5)

    def test_vec3_quat_to_mat44(self):
        pool = TransformBufferPool()
        source = _make_source_vec3_quat(n=1)

        result = pool.get_or_convert(
            source,
            TransformFormat.MAT44,
            generation=1,
            matrix_layout=MatrixLayout.ROW_MAJOR,
        )
        assert isinstance(result, Mat44Transforms)
        wp.synchronize_device(DEVICE)

        m = result.matrices.numpy()[0]
        np.testing.assert_allclose(m[0, 3], 1.0, atol=1e-5)
        np.testing.assert_allclose(m[1, 3], 2.0, atol=1e-5)
        np.testing.assert_allclose(m[2, 3], 3.0, atol=1e-5)

    def test_vec3_quat_to_mat44_column_major(self):
        pool = TransformBufferPool()
        source = _make_source_vec3_quat(n=1)

        result = pool.get_or_convert(
            source,
            TransformFormat.MAT44,
            generation=1,
            matrix_layout=MatrixLayout.COLUMN_MAJOR,
        )
        assert isinstance(result, Mat44Transforms)
        assert result.layout == MatrixLayout.COLUMN_MAJOR
        wp.synchronize_device(DEVICE)

        m = result.matrices.numpy()[0]
        # Column-major: translation is in row 3 (transposed)
        np.testing.assert_allclose(m[3, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(m[3, 1], 2.0, atol=1e-5)
        np.testing.assert_allclose(m[3, 2], 3.0, atol=1e-5)

    def test_transform_to_vec3_quat(self):
        pool = TransformBufferPool()
        source = _make_source_transform(n=2)

        result = pool.get_or_convert(source, TransformFormat.VEC3_QUAT, generation=1)
        assert isinstance(result, Vec3QuatTransforms)
        assert result.count == 2
        wp.synchronize_device(DEVICE)

        pos = result.positions.numpy()
        np.testing.assert_allclose(pos[0], [1.0, 2.0, 3.0], atol=1e-5)


class TestClear:
    def test_clear_releases_cache(self):
        pool = TransformBufferPool()
        source = _make_source_vec3_quat()

        pool.get_or_convert(source, TransformFormat.TRANSFORM, generation=1)
        assert len(pool._cache) > 0

        pool.clear()
        assert len(pool._cache) == 0
        assert len(pool._cache_generation) == 0
