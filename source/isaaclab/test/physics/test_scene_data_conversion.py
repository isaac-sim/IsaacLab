# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ConversionDispatcher."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from isaaclab.physics.scene_data_conversion import ConversionDispatcher
from isaaclab.physics.scene_data_types import (
    Mat44Transforms,
    MatrixLayout,
    QuaternionConvention,
    TransformArrayData,
    TransformFormat,
    Vec3Mat33Transforms,
    Vec3QuatTransforms,
)

DEVICE = "cuda:0"
ATOL = 1e-5


def _identity_quat_np():
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _z90_quat_np():
    angle = math.pi / 2
    return np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)], dtype=np.float32)


def _make_vec3_quat(n=2):
    positions = wp.from_numpy(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]][:n], dtype=np.float32),
        dtype=wp.vec3,
        device=DEVICE,
    )
    quats = wp.from_numpy(
        np.array([_identity_quat_np(), _z90_quat_np()][:n], dtype=np.float32),
        dtype=wp.quatf,
        device=DEVICE,
    )
    return Vec3QuatTransforms(count=n, device=DEVICE, positions=positions, orientations=quats)


def _make_transform(n=2):
    tf_np = np.zeros((n, 7), dtype=np.float32)
    tf_np[0, :3] = [1.0, 2.0, 3.0]
    tf_np[0, 3:7] = _identity_quat_np()
    if n > 1:
        tf_np[1, :3] = [4.0, 5.0, 6.0]
        tf_np[1, 3:7] = _z90_quat_np()
    return TransformArrayData(
        count=n,
        device=DEVICE,
        transforms=wp.from_numpy(tf_np, dtype=wp.transformf, device=DEVICE),
    )


def _alloc_vec3_quat(n=2):
    return Vec3QuatTransforms(
        count=n,
        device=DEVICE,
        positions=wp.zeros(n, dtype=wp.vec3, device=DEVICE),
        orientations=wp.zeros(n, dtype=wp.quatf, device=DEVICE),
    )


def _alloc_transform(n=2):
    return TransformArrayData(
        count=n,
        device=DEVICE,
        transforms=wp.zeros(n, dtype=wp.transformf, device=DEVICE),
    )


def _alloc_mat44f(n=2, layout=MatrixLayout.ROW_MAJOR):
    return Mat44Transforms(
        count=n,
        device=DEVICE,
        matrices=wp.zeros(n, dtype=wp.mat44f, device=DEVICE),
        layout=layout,
    )


def _alloc_vec3_mat33(n=2):
    return Vec3Mat33Transforms(
        count=n,
        device=DEVICE,
        positions=wp.zeros(n, dtype=wp.vec3, device=DEVICE),
        rotations=wp.zeros(n, dtype=wp.mat33f, device=DEVICE),
    )


class TestCanPassthrough:
    def test_same_format_same_convention(self):
        source = _make_vec3_quat()
        assert ConversionDispatcher.can_passthrough(source, TransformFormat.VEC3_QUAT)

    def test_same_format_different_convention(self):
        source = _make_vec3_quat()
        assert not ConversionDispatcher.can_passthrough(
            source, TransformFormat.VEC3_QUAT, quat_convention=QuaternionConvention.WXYZ
        )

    def test_different_format(self):
        source = _make_vec3_quat()
        assert not ConversionDispatcher.can_passthrough(source, TransformFormat.TRANSFORM)

    def test_transform_passthrough(self):
        source = _make_transform()
        assert ConversionDispatcher.can_passthrough(source, TransformFormat.TRANSFORM)

    def test_mat44_layout_mismatch(self):
        source = Mat44Transforms(
            count=1,
            device=DEVICE,
            matrices=wp.zeros(1, dtype=wp.mat44f, device=DEVICE),
            layout=MatrixLayout.ROW_MAJOR,
        )
        assert not ConversionDispatcher.can_passthrough(
            source, TransformFormat.MAT44, matrix_layout=MatrixLayout.COLUMN_MAJOR
        )


class TestConvertVec3QuatToTransform:
    def test_positions_preserved(self):
        source = _make_vec3_quat()
        target = _alloc_transform()

        ConversionDispatcher.convert(source, target)
        wp.synchronize_device(DEVICE)

        tf = target.transforms.numpy()
        np.testing.assert_allclose(tf[0][:3], [1.0, 2.0, 3.0], atol=ATOL)
        np.testing.assert_allclose(tf[1][:3], [4.0, 5.0, 6.0], atol=ATOL)


class TestConvertTransformToVec3Quat:
    def test_roundtrip(self):
        """transform -> vec3_quat -> transform should recover original."""
        source = _make_transform()
        mid = _alloc_vec3_quat()
        ConversionDispatcher.convert(source, mid)

        final = _alloc_transform()
        ConversionDispatcher.convert(mid, final)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(final.transforms.numpy(), source.transforms.numpy(), atol=ATOL)


class TestConvertVec3QuatToMat44:
    def test_identity_produces_identity(self):
        source = Vec3QuatTransforms(
            count=1,
            device=DEVICE,
            positions=wp.from_numpy(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=DEVICE),
            orientations=wp.from_numpy(
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.quatf, device=DEVICE
            ),
        )
        target = _alloc_mat44f(1)
        ConversionDispatcher.convert(source, target)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(target.matrices.numpy()[0], np.eye(4, dtype=np.float32), atol=ATOL)


class TestConvertVec3QuatToVec3Mat33:
    def test_identity_quat_gives_identity_rotation(self):
        source = Vec3QuatTransforms(
            count=1,
            device=DEVICE,
            positions=wp.from_numpy(np.array([[5.0, 6.0, 7.0]], dtype=np.float32), dtype=wp.vec3, device=DEVICE),
            orientations=wp.from_numpy(
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.quatf, device=DEVICE
            ),
        )
        target = _alloc_vec3_mat33(1)
        ConversionDispatcher.convert(source, target)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(target.positions.numpy()[0], [5.0, 6.0, 7.0], atol=ATOL)
        np.testing.assert_allclose(target.rotations.numpy()[0], np.eye(3, dtype=np.float32), atol=ATOL)


class TestConvertWithIndexMap:
    def test_scatter_preserves_data(self):
        source = _make_vec3_quat(2)
        index_map = wp.from_numpy(np.array([3, 1], dtype=np.int32), device=DEVICE)
        target = _alloc_transform(5)

        ConversionDispatcher.convert(source, target, index_map=index_map)
        wp.synchronize_device(DEVICE)

        tf = target.transforms.numpy()
        np.testing.assert_allclose(tf[3][:3], [1.0, 2.0, 3.0], atol=ATOL)
        np.testing.assert_allclose(tf[1][:3], [4.0, 5.0, 6.0], atol=ATOL)
        np.testing.assert_allclose(tf[0][:3], [0.0, 0.0, 0.0], atol=ATOL)


class TestQuatConventionSwizzle:
    def test_convert_to_wxyz(self):
        source = _make_vec3_quat(1)
        target = Vec3QuatTransforms(
            count=1,
            device=DEVICE,
            positions=wp.zeros(1, dtype=wp.vec3, device=DEVICE),
            orientations=wp.zeros(1, dtype=wp.quatf, device=DEVICE),
            quat_convention=QuaternionConvention.WXYZ,
        )
        ConversionDispatcher.convert(source, target)
        wp.synchronize_device(DEVICE)

        result = target.orientations.numpy()[0]
        # Source was identity XYZW: (0,0,0,1). WXYZ should be (1,0,0,0).
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=ATOL)


class TestMatrixLayoutTranspose:
    def test_column_major_output(self):
        source = _make_vec3_quat(1)
        target = Mat44Transforms(
            count=1,
            device=DEVICE,
            matrices=wp.zeros(1, dtype=wp.mat44f, device=DEVICE),
            layout=MatrixLayout.COLUMN_MAJOR,
        )
        ConversionDispatcher.convert(source, target)
        wp.synchronize_device(DEVICE)

        m = target.matrices.numpy()[0]
        # Column-major identity with translation (1,2,3):
        # translation should be in row 3 (transposed)
        np.testing.assert_allclose(m[3, 0], 1.0, atol=ATOL)
        np.testing.assert_allclose(m[3, 1], 2.0, atol=ATOL)
        np.testing.assert_allclose(m[3, 2], 3.0, atol=ATOL)


class TestBaseProviderDefaults:
    def test_get_body_transforms_default_returns_none(self):
        from isaaclab.physics.base_scene_data_provider import BaseSceneDataProvider

        class Stub(BaseSceneDataProvider):
            def update(self, env_ids=None):
                pass

            def get_newton_model(self):
                return None

            def get_newton_state(self, env_ids=None):
                return None

            def get_usd_stage(self):
                return None

            def get_metadata(self):
                return {}

            def get_transforms(self):
                return None

            def get_velocities(self):
                return None

            def get_contacts(self):
                return None

            def get_camera_transforms(self):
                return None

        provider = Stub()
        assert provider.get_body_transforms(TransformFormat.TRANSFORM) is None
        assert provider.get_source_format() is None
