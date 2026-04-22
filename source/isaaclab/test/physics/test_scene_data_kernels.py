# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for scene data conversion kernels.

These tests verify all 16 conversion paths in the kernel grid, plus
quaternion convention swizzle, matrix transpose, and index-mapped
subset scatter. All tests run on GPU via Warp and do not require
Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from isaaclab.physics import scene_data_kernels as K

DEVICE = "cuda:0"
ATOL = 1e-5


def _identity_quat():
    """Return an XYZW identity quaternion as numpy: (0, 0, 0, 1)."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _rotation_quat_z90():
    """Return a 90-degree rotation about Z as XYZW quaternion."""
    angle = math.pi / 2
    return np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)], dtype=np.float32)


def _make_test_data(n=4):
    """Create test positions and XYZW quaternions as numpy arrays."""
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.1, 0.2, 0.3]], dtype=np.float32)[:n]
    quats = np.array([_identity_quat(), _rotation_quat_z90(), _identity_quat(), _rotation_quat_z90()],
                     dtype=np.float32)[:n]
    return positions, quats


def _to_warp_arrays(positions_np, quats_np, device=DEVICE):
    """Convert numpy positions and quaternions to Warp arrays."""
    pos_wp = wp.from_numpy(positions_np.copy(), dtype=wp.vec3, device=device)
    ori_wp = wp.from_numpy(quats_np.copy(), dtype=wp.quatf, device=device)
    return pos_wp, ori_wp


def _empty_imap(device=DEVICE):
    return wp.empty(0, dtype=wp.int32, device=device)


class TestVec3QuatToTransform:
    def test_basic_conversion(self):
        positions_np, quats_np = _make_test_data(2)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)
        dst = wp.zeros(2, dtype=wp.transformf, device=DEVICE)

        wp.launch(K.vec3_quat_to_transform_kernel, dim=2, inputs=[pos_wp, ori_wp, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        result = dst.numpy()
        for i in range(2):
            np.testing.assert_allclose(result[i][:3], positions_np[i], atol=ATOL)
            np.testing.assert_allclose(result[i][3:7], quats_np[i], atol=ATOL)


class TestTransformToVec3Quat:
    def test_roundtrip(self):
        """vec3_quat -> transform -> vec3_quat should recover original data."""
        positions_np, quats_np = _make_test_data(4)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        tf = wp.zeros(4, dtype=wp.transformf, device=DEVICE)
        wp.launch(K.vec3_quat_to_transform_kernel, dim=4, inputs=[pos_wp, ori_wp, tf, _empty_imap(), 0], device=DEVICE)

        out_pos = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        out_ori = wp.zeros(4, dtype=wp.quatf, device=DEVICE)
        wp.launch(K.transform_to_vec3_quat_kernel, dim=4, inputs=[tf, out_pos, out_ori, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(out_pos.numpy(), positions_np, atol=ATOL)
        np.testing.assert_allclose(np.abs(out_ori.numpy()), np.abs(quats_np), atol=ATOL)


class TestVec3QuatToVec3Mat33:
    def test_identity_quat_gives_identity_matrix(self):
        pos_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        quat_np = np.array([_identity_quat()], dtype=np.float32)
        pos_wp, ori_wp = _to_warp_arrays(pos_np, quat_np)

        dst_pos = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        dst_rot = wp.zeros(1, dtype=wp.mat33f, device=DEVICE)
        wp.launch(
            K.vec3_quat_to_vec3_mat33_kernel,
            dim=1,
            inputs=[pos_wp, ori_wp, dst_pos, dst_rot, _empty_imap(), 0],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(dst_pos.numpy()[0], [1.0, 2.0, 3.0], atol=ATOL)
        np.testing.assert_allclose(dst_rot.numpy()[0], np.eye(3, dtype=np.float32), atol=ATOL)

    def test_roundtrip(self):
        """vec3_quat -> vec3_mat33 -> vec3_quat should recover original data."""
        positions_np, quats_np = _make_test_data(2)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        mid_pos = wp.zeros(2, dtype=wp.vec3, device=DEVICE)
        mid_rot = wp.zeros(2, dtype=wp.mat33f, device=DEVICE)
        wp.launch(
            K.vec3_quat_to_vec3_mat33_kernel,
            dim=2,
            inputs=[pos_wp, ori_wp, mid_pos, mid_rot, _empty_imap(), 0],
            device=DEVICE,
        )

        out_pos = wp.zeros(2, dtype=wp.vec3, device=DEVICE)
        out_ori = wp.zeros(2, dtype=wp.quatf, device=DEVICE)
        wp.launch(
            K.vec3_mat33_to_vec3_quat_kernel,
            dim=2,
            inputs=[mid_pos, mid_rot, out_pos, out_ori, _empty_imap(), 0],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(out_pos.numpy(), positions_np, atol=ATOL)
        np.testing.assert_allclose(np.abs(out_ori.numpy()), np.abs(quats_np), atol=ATOL)


class TestVec3QuatToMat44:
    def test_identity_produces_identity_matrix(self):
        pos_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        quat_np = np.array([_identity_quat()], dtype=np.float32)
        pos_wp, ori_wp = _to_warp_arrays(pos_np, quat_np)

        dst = wp.zeros(1, dtype=wp.mat44f, device=DEVICE)
        wp.launch(K.vec3_quat_to_mat44f_kernel, dim=1, inputs=[pos_wp, ori_wp, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(dst.numpy()[0], np.eye(4, dtype=np.float32), atol=ATOL)

    def test_translation_in_last_column(self):
        pos_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        quat_np = np.array([_identity_quat()], dtype=np.float32)
        pos_wp, ori_wp = _to_warp_arrays(pos_np, quat_np)

        dst = wp.zeros(1, dtype=wp.mat44f, device=DEVICE)
        wp.launch(K.vec3_quat_to_mat44f_kernel, dim=1, inputs=[pos_wp, ori_wp, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        m = dst.numpy()[0]
        np.testing.assert_allclose(m[0, 3], 10.0, atol=ATOL)
        np.testing.assert_allclose(m[1, 3], 20.0, atol=ATOL)
        np.testing.assert_allclose(m[2, 3], 30.0, atol=ATOL)

    def test_double_precision(self):
        pos_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        quat_np = np.array([_identity_quat()], dtype=np.float32)
        pos_wp, ori_wp = _to_warp_arrays(pos_np, quat_np)

        dst = wp.zeros(1, dtype=wp.mat44d, device=DEVICE)
        wp.launch(K.vec3_quat_to_mat44d_kernel, dim=1, inputs=[pos_wp, ori_wp, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        m = dst.numpy()[0]
        np.testing.assert_allclose(m[0, 3], 1.0, atol=1e-10)


class TestMat44ToVec3Quat:
    def test_roundtrip_float32(self):
        """vec3_quat -> mat44f -> vec3_quat should recover original data."""
        positions_np, quats_np = _make_test_data(2)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        mat = wp.zeros(2, dtype=wp.mat44f, device=DEVICE)
        wp.launch(K.vec3_quat_to_mat44f_kernel, dim=2, inputs=[pos_wp, ori_wp, mat, _empty_imap(), 0], device=DEVICE)

        out_pos = wp.zeros(2, dtype=wp.vec3, device=DEVICE)
        out_ori = wp.zeros(2, dtype=wp.quatf, device=DEVICE)
        wp.launch(K.mat44f_to_vec3_quat_kernel, dim=2, inputs=[mat, out_pos, out_ori, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(out_pos.numpy(), positions_np, atol=ATOL)
        np.testing.assert_allclose(np.abs(out_ori.numpy()), np.abs(quats_np), atol=ATOL)

    def test_roundtrip_float64(self):
        """vec3_quat -> mat44d -> vec3_quat should recover original data."""
        positions_np, quats_np = _make_test_data(2)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        mat = wp.zeros(2, dtype=wp.mat44d, device=DEVICE)
        wp.launch(K.vec3_quat_to_mat44d_kernel, dim=2, inputs=[pos_wp, ori_wp, mat, _empty_imap(), 0], device=DEVICE)

        out_pos = wp.zeros(2, dtype=wp.vec3, device=DEVICE)
        out_ori = wp.zeros(2, dtype=wp.quatf, device=DEVICE)
        wp.launch(K.mat44d_to_vec3_quat_kernel, dim=2, inputs=[mat, out_pos, out_ori, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(out_pos.numpy(), positions_np, atol=ATOL)
        np.testing.assert_allclose(np.abs(out_ori.numpy()), np.abs(quats_np), atol=ATOL)


class TestTransformToMat44:
    def test_roundtrip(self):
        """transform -> mat44f -> transform should recover original data."""
        positions_np, quats_np = _make_test_data(3)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        tf = wp.zeros(3, dtype=wp.transformf, device=DEVICE)
        wp.launch(K.vec3_quat_to_transform_kernel, dim=3, inputs=[pos_wp, ori_wp, tf, _empty_imap(), 0], device=DEVICE)

        mat = wp.zeros(3, dtype=wp.mat44f, device=DEVICE)
        wp.launch(K.transform_to_mat44f_kernel, dim=3, inputs=[tf, mat, _empty_imap(), 0], device=DEVICE)

        tf2 = wp.zeros(3, dtype=wp.transformf, device=DEVICE)
        wp.launch(K.mat44f_to_transform_kernel, dim=3, inputs=[mat, tf2, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(tf2.numpy(), tf.numpy(), atol=ATOL)


class TestIndexMap:
    def test_scatter_write(self):
        """Index map should scatter source elements to specified output positions."""
        positions_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        quats_np = np.array([_identity_quat(), _rotation_quat_z90()], dtype=np.float32)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        index_map = wp.from_numpy(np.array([2, 0], dtype=np.int32), device=DEVICE)

        dst_pos = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        dst_ori = wp.zeros(4, dtype=wp.quatf, device=DEVICE)

        wp.launch(
            K.vec3_quat_to_vec3_quat_kernel,
            dim=2,
            inputs=[pos_wp, ori_wp, dst_pos, dst_ori, index_map, 2],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        result_pos = dst_pos.numpy()
        np.testing.assert_allclose(result_pos[2], [1.0, 2.0, 3.0], atol=ATOL)
        np.testing.assert_allclose(result_pos[0], [4.0, 5.0, 6.0], atol=ATOL)
        np.testing.assert_allclose(result_pos[1], [0.0, 0.0, 0.0], atol=ATOL)
        np.testing.assert_allclose(result_pos[3], [0.0, 0.0, 0.0], atol=ATOL)

    def test_index_map_with_transform_conversion(self):
        """Index map should work with format conversion (vec3_quat -> transform)."""
        positions_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        quats_np = np.array([_identity_quat()], dtype=np.float32)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)

        index_map = wp.from_numpy(np.array([3], dtype=np.int32), device=DEVICE)
        dst = wp.zeros(5, dtype=wp.transformf, device=DEVICE)

        wp.launch(
            K.vec3_quat_to_transform_kernel,
            dim=1,
            inputs=[pos_wp, ori_wp, dst, index_map, 1],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)

        result = dst.numpy()
        np.testing.assert_allclose(result[3][:3], [10.0, 20.0, 30.0], atol=ATOL)
        np.testing.assert_allclose(result[0][:3], [0.0, 0.0, 0.0], atol=ATOL)


class TestQuaternionSwizzle:
    def test_xyzw_to_wxyz(self):
        quat_xyzw = np.array([[0.1, 0.2, 0.3, 0.9]], dtype=np.float32)
        src = wp.from_numpy(quat_xyzw, dtype=wp.quatf, device=DEVICE)
        dst = wp.zeros(1, dtype=wp.quatf, device=DEVICE)

        wp.launch(K.quat_xyzw_to_wxyz_kernel, dim=1, inputs=[src, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        result = dst.numpy()[0]
        np.testing.assert_allclose(result, [0.9, 0.1, 0.2, 0.3], atol=ATOL)

    def test_roundtrip(self):
        """XYZW -> WXYZ -> XYZW should recover original data."""
        quat_xyzw = np.array([[0.1, 0.2, 0.3, 0.9]], dtype=np.float32)
        src = wp.from_numpy(quat_xyzw, dtype=wp.quatf, device=DEVICE)
        mid = wp.zeros(1, dtype=wp.quatf, device=DEVICE)
        dst = wp.zeros(1, dtype=wp.quatf, device=DEVICE)

        wp.launch(K.quat_xyzw_to_wxyz_kernel, dim=1, inputs=[src, mid, _empty_imap(), 0], device=DEVICE)
        wp.launch(K.quat_wxyz_to_xyzw_kernel, dim=1, inputs=[mid, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(dst.numpy()[0], quat_xyzw[0], atol=ATOL)


class TestMatrixTranspose:
    def test_mat44f_transpose_roundtrip(self):
        m_np = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.float32
        )
        src = wp.from_numpy(m_np, dtype=wp.mat44f, device=DEVICE)
        mid = wp.zeros(1, dtype=wp.mat44f, device=DEVICE)
        dst = wp.zeros(1, dtype=wp.mat44f, device=DEVICE)

        wp.launch(K.mat44f_transpose_kernel, dim=1, inputs=[src, mid, _empty_imap(), 0], device=DEVICE)
        wp.launch(K.mat44f_transpose_kernel, dim=1, inputs=[mid, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(dst.numpy(), m_np, atol=ATOL)

    def test_mat44f_transpose_values(self):
        m_np = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.float32
        )
        src = wp.from_numpy(m_np, dtype=wp.mat44f, device=DEVICE)
        dst = wp.zeros(1, dtype=wp.mat44f, device=DEVICE)

        wp.launch(K.mat44f_transpose_kernel, dim=1, inputs=[src, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        expected = m_np[0].T
        np.testing.assert_allclose(dst.numpy()[0], expected, atol=ATOL)


class TestEdgeCases:
    def test_single_element(self):
        positions_np, quats_np = _make_test_data(1)
        pos_wp, ori_wp = _to_warp_arrays(positions_np, quats_np)
        dst = wp.zeros(1, dtype=wp.transformf, device=DEVICE)

        wp.launch(K.vec3_quat_to_transform_kernel, dim=1, inputs=[pos_wp, ori_wp, dst, _empty_imap(), 0], device=DEVICE)
        wp.synchronize_device(DEVICE)

        result = dst.numpy()
        np.testing.assert_allclose(result[0][:3], positions_np[0], atol=ATOL)
