# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for scene data types."""

from __future__ import annotations

import pytest
import warp as wp

from isaaclab.physics.scene_data_types import (
    Mat44Transforms,
    MatrixLayout,
    QuaternionConvention,
    TransformArrayData,
    TransformData,
    TransformFormat,
    Vec3Mat33Transforms,
    Vec3QuatTransforms,
)

DEVICE = "cuda:0"


class TestEnums:
    def test_quaternion_convention_values(self):
        assert QuaternionConvention.XYZW.value == "xyzw"
        assert QuaternionConvention.WXYZ.value == "wxyz"

    def test_matrix_layout_values(self):
        assert MatrixLayout.ROW_MAJOR.value == "row_major"
        assert MatrixLayout.COLUMN_MAJOR.value == "column_major"

    def test_transform_format_values(self):
        assert TransformFormat.VEC3_QUAT.value == "vec3_quat"
        assert TransformFormat.VEC3_MAT33.value == "vec3_mat33"
        assert TransformFormat.TRANSFORM.value == "transform"
        assert TransformFormat.MAT44.value == "mat44"


class TestVec3QuatTransforms:
    def test_construction(self):
        n = 10
        t = Vec3QuatTransforms(
            count=n,
            device=DEVICE,
            positions=wp.zeros(n, dtype=wp.vec3, device=DEVICE),
            orientations=wp.zeros(n, dtype=wp.quatf, device=DEVICE),
        )
        assert t.format == TransformFormat.VEC3_QUAT
        assert t.count == n
        assert t.device == DEVICE
        assert t.quat_convention == QuaternionConvention.XYZW

    def test_wxyz_convention(self):
        t = Vec3QuatTransforms(
            count=1,
            device=DEVICE,
            positions=wp.zeros(1, dtype=wp.vec3, device=DEVICE),
            orientations=wp.zeros(1, dtype=wp.quatf, device=DEVICE),
            quat_convention=QuaternionConvention.WXYZ,
        )
        assert t.quat_convention == QuaternionConvention.WXYZ

    def test_inherits_transform_data(self):
        t = Vec3QuatTransforms(count=1, device=DEVICE)
        assert isinstance(t, TransformData)


class TestVec3Mat33Transforms:
    def test_construction(self):
        n = 5
        t = Vec3Mat33Transforms(
            count=n,
            device=DEVICE,
            positions=wp.zeros(n, dtype=wp.vec3, device=DEVICE),
            rotations=wp.zeros(n, dtype=wp.mat33f, device=DEVICE),
        )
        assert t.format == TransformFormat.VEC3_MAT33
        assert t.layout == MatrixLayout.ROW_MAJOR

    def test_column_major(self):
        t = Vec3Mat33Transforms(count=1, device=DEVICE, layout=MatrixLayout.COLUMN_MAJOR)
        assert t.layout == MatrixLayout.COLUMN_MAJOR


class TestTransformArrayData:
    def test_construction(self):
        n = 8
        t = TransformArrayData(
            count=n,
            device=DEVICE,
            transforms=wp.zeros(n, dtype=wp.transformf, device=DEVICE),
        )
        assert t.format == TransformFormat.TRANSFORM
        assert t.count == n


class TestMat44Transforms:
    def test_float32(self):
        n = 3
        t = Mat44Transforms(
            count=n,
            device=DEVICE,
            matrices=wp.zeros(n, dtype=wp.mat44f, device=DEVICE),
        )
        assert t.format == TransformFormat.MAT44
        assert t.double_precision is False
        assert t.layout == MatrixLayout.COLUMN_MAJOR

    def test_float64(self):
        n = 3
        t = Mat44Transforms(
            count=n,
            device=DEVICE,
            matrices=wp.zeros(n, dtype=wp.mat44d, device=DEVICE),
            double_precision=True,
        )
        assert t.double_precision is True

    def test_row_major(self):
        t = Mat44Transforms(count=1, device=DEVICE, layout=MatrixLayout.ROW_MAJOR)
        assert t.layout == MatrixLayout.ROW_MAJOR
