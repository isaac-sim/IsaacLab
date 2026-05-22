# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Warp fabric transform kernels.

Tests the decompose/compose math via helper kernels that operate on
regular wp.array (no Fabric/USDRT runtime required).
"""

import numpy as np
import warp as wp

wp.init()

from isaaclab.utils.warp.fabric import _decompose_transformation_matrix  # noqa: E402


@wp.kernel(enable_backward=False)
def _test_decompose_kernel(
    matrices: wp.array(dtype=wp.mat44f),
    out_positions: wp.array(dtype=wp.vec3f),
    out_rotations: wp.array(dtype=wp.vec4f),
    out_scales: wp.array(dtype=wp.vec3f),
):
    """Decompose a batch of 4x4 matrices into pos/quat/scale."""
    i = wp.tid()
    pos, rot, scale = _decompose_transformation_matrix(matrices[i])
    out_positions[i] = pos
    out_rotations[i] = wp.vec4f(rot[0], rot[1], rot[2], rot[3])
    out_scales[i] = scale


@wp.kernel(enable_backward=False)
def _test_compose_kernel(
    positions: wp.array(dtype=wp.vec3f),
    rotations: wp.array(dtype=wp.vec4f),
    scales: wp.array(dtype=wp.vec3f),
    out_matrices: wp.array(dtype=wp.mat44f),
):
    """Compose a batch of pos/quat/scale into 4x4 matrices."""
    i = wp.tid()
    pos = positions[i]
    rot = wp.quatf(rotations[i][0], rotations[i][1], rotations[i][2], rotations[i][3])
    scale = scales[i]
    out_matrices[i] = wp.transpose(wp.transform_compose(pos, rot, scale))


class TestDecomposeCompose:
    """Round-trip tests for decompose ↔ compose transform math."""

    def test_identity_matrix(self):
        """Identity matrix decomposes to pos=0, quat=identity, scale=1."""
        mat = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        matrices = wp.array(mat, dtype=wp.mat44f, device="cpu")
        out_pos = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        out_rot = wp.zeros(1, dtype=wp.vec4f, device="cpu")
        out_scale = wp.zeros(1, dtype=wp.vec3f, device="cpu")

        wp.launch(_test_decompose_kernel, dim=1, inputs=[matrices, out_pos, out_rot, out_scale], device="cpu")
        wp.synchronize()

        pos = out_pos.numpy()
        rot = out_rot.numpy()
        scale = out_scale.numpy()

        np.testing.assert_allclose(pos[0], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(scale[0], [1, 1, 1], atol=1e-6)
        # Identity quaternion: either (0,0,0,1) or (0,0,0,-1)
        assert abs(abs(rot[0, 3]) - 1.0) < 1e-5

    def test_translation_only(self):
        """Matrix with only translation decomposes correctly."""
        mat = np.eye(4, dtype=np.float32)
        mat[3, 0] = 1.0  # row-major: translation in last row
        mat[3, 1] = 2.0
        mat[3, 2] = 3.0
        matrices = wp.array(mat.reshape(1, 4, 4), dtype=wp.mat44f, device="cpu")
        out_pos = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        out_rot = wp.zeros(1, dtype=wp.vec4f, device="cpu")
        out_scale = wp.zeros(1, dtype=wp.vec3f, device="cpu")

        wp.launch(_test_decompose_kernel, dim=1, inputs=[matrices, out_pos, out_rot, out_scale], device="cpu")
        wp.synchronize()

        np.testing.assert_allclose(out_pos.numpy()[0], [1, 2, 3], atol=1e-6)
        np.testing.assert_allclose(out_scale.numpy()[0], [1, 1, 1], atol=1e-6)

    def test_uniform_scale(self):
        """Matrix with uniform scale decomposes correctly."""
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = 2.0
        mat[1, 1] = 2.0
        mat[2, 2] = 2.0
        matrices = wp.array(mat.reshape(1, 4, 4), dtype=wp.mat44f, device="cpu")
        out_pos = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        out_rot = wp.zeros(1, dtype=wp.vec4f, device="cpu")
        out_scale = wp.zeros(1, dtype=wp.vec3f, device="cpu")

        wp.launch(_test_decompose_kernel, dim=1, inputs=[matrices, out_pos, out_rot, out_scale], device="cpu")
        wp.synchronize()

        np.testing.assert_allclose(out_scale.numpy()[0], [2, 2, 2], atol=1e-6)

    def test_round_trip(self):
        """Compose then decompose recovers original pos/quat/scale."""
        # Known transform: translate (5,6,7), rotate 90° about Z, scale (1,2,3)
        pos = np.array([[5.0, 6.0, 7.0]], dtype=np.float32)
        # 90° about Z in xyzw: (0, 0, sin(45°), cos(45°))
        s45 = np.sin(np.pi / 4)
        c45 = np.cos(np.pi / 4)
        rot = np.array([[0.0, 0.0, s45, c45]], dtype=np.float32)
        scale = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        wp_pos = wp.array(pos, dtype=wp.vec3f, device="cpu")
        wp_rot = wp.array(rot, dtype=wp.vec4f, device="cpu")
        wp_scale = wp.array(scale, dtype=wp.vec3f, device="cpu")
        out_mat = wp.zeros(1, dtype=wp.mat44f, device="cpu")

        # Compose
        wp.launch(_test_compose_kernel, dim=1, inputs=[wp_pos, wp_rot, wp_scale, out_mat], device="cpu")
        wp.synchronize()

        # Decompose
        out_pos = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        out_rot = wp.zeros(1, dtype=wp.vec4f, device="cpu")
        out_scale = wp.zeros(1, dtype=wp.vec3f, device="cpu")

        wp.launch(_test_decompose_kernel, dim=1, inputs=[out_mat, out_pos, out_rot, out_scale], device="cpu")
        wp.synchronize()

        np.testing.assert_allclose(out_pos.numpy()[0], pos[0], atol=1e-5)
        np.testing.assert_allclose(out_scale.numpy()[0], scale[0], atol=1e-5)
        # Quaternion sign ambiguity
        r_out = out_rot.numpy()[0]
        r_exp = rot[0]
        dot = np.dot(r_out, r_exp)
        np.testing.assert_allclose(abs(dot), 1.0, atol=1e-5)

    def test_non_uniform_scale_round_trip(self):
        """Non-uniform scale round-trips correctly."""
        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        rot = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)  # identity
        scale = np.array([[0.5, 2.0, 3.0]], dtype=np.float32)

        wp_pos = wp.array(pos, dtype=wp.vec3f, device="cpu")
        wp_rot = wp.array(rot, dtype=wp.vec4f, device="cpu")
        wp_scale = wp.array(scale, dtype=wp.vec3f, device="cpu")
        out_mat = wp.zeros(1, dtype=wp.mat44f, device="cpu")

        wp.launch(_test_compose_kernel, dim=1, inputs=[wp_pos, wp_rot, wp_scale, out_mat], device="cpu")
        wp.synchronize()

        out_pos = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        out_rot = wp.zeros(1, dtype=wp.vec4f, device="cpu")
        out_scale = wp.zeros(1, dtype=wp.vec3f, device="cpu")

        wp.launch(_test_decompose_kernel, dim=1, inputs=[out_mat, out_pos, out_rot, out_scale], device="cpu")
        wp.synchronize()

        np.testing.assert_allclose(out_scale.numpy()[0], scale[0], atol=1e-5)


class TestKernelSignatures:
    """Verify all exported kernels are importable and are Warp Kernels."""

    def test_all_kernels_importable(self):
        """All public kernels listed in __all__ should be importable and be Warp Kernels."""
        from isaaclab.utils.warp import fabric as fabric_utils

        expected_kernels = [
            "arange_k",
            "compose_fabric_transformation_matrix_from_warp_arrays",
            "compose_indexed_fabric_transforms",
            "decompose_fabric_transformation_matrix_to_warp_arrays",
            "decompose_indexed_fabric_transforms",
            "set_view_to_fabric_array",
            "update_indexed_local_matrix_from_world",
            "update_indexed_world_matrix_from_local",
        ]

        for name in expected_kernels:
            obj = getattr(fabric_utils, name, None)
            assert obj is not None, f"{name} not found in fabric_utils"
            assert isinstance(obj, wp.Kernel), f"{name} should be a wp.Kernel, got {type(obj)}"

    def test_module_exports_match_all(self):
        """__all__ should list every public kernel."""
        from isaaclab.utils.warp import fabric as fabric_utils

        for name in fabric_utils.__all__:
            assert hasattr(fabric_utils, name), f"__all__ lists '{name}' but it's not defined"
