# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Warp fabric transform kernels.

Tests the shared @wp.func math (decompose/compose and matrix inverse/multiply)
through plain wp.array kernels — no Fabric/USDRT runtime required.

The production fabric kernels are thin adapters over the same math; testing the
math in isolation avoids coupling tests to Fabric container internals.
"""

import numpy as np
import warp as wp

wp.init()

from isaaclab.utils.warp.fabric import _decompose_transformation_matrix  # noqa: E402


# ------------------------------------------------------------------
# Test kernels (wp.array wrappers around the same math as production)
# ------------------------------------------------------------------


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


@wp.kernel(enable_backward=False)
def _test_local_from_world_kernel(
    child_world: wp.array(dtype=wp.mat44d),
    parent_world: wp.array(dtype=wp.mat44d),
    out_local: wp.array(dtype=wp.mat44d),
):
    """Same math as update_indexed_local_matrix_from_world: local^T = world^T * inv(parent^T).

    Casts to mat44f for compute (matching production precision), writes back as mat44d.
    """
    i = wp.tid()
    cw = wp.mat44f(child_world[i])
    pw = wp.mat44f(parent_world[i])
    out_local[i] = wp.mat44d(cw * wp.inverse(pw))


@wp.kernel(enable_backward=False)
def _test_world_from_local_kernel(
    child_local: wp.array(dtype=wp.mat44d),
    parent_world: wp.array(dtype=wp.mat44d),
    out_world: wp.array(dtype=wp.mat44d),
):
    """Same math as update_indexed_world_matrix_from_local: world^T = local^T * parent^T."""
    i = wp.tid()
    cl = wp.mat44f(child_local[i])
    pw = wp.mat44f(parent_world[i])
    out_world[i] = wp.mat44d(cl * pw)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_transform_matrix(pos, rot_quat_xyzw, scale):
    """Build a 4x4 Fabric-transposed transform from pos/quat/scale.

    Returns numpy (4,4) float64 in the transposed storage convention (row-major with
    translation in the last row) that Fabric uses.
    """
    from scipy.spatial.transform import Rotation

    r = Rotation.from_quat(rot_quat_xyzw).as_matrix().astype(np.float64)
    rs = r * np.array(scale, dtype=np.float64)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rs
    m[:3, 3] = pos
    # Transpose for Fabric storage convention
    return m.T


# ------------------------------------------------------------------
# Decompose / Compose round-trip tests
# ------------------------------------------------------------------


def test_identity_matrix():
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


def test_translation_only():
    """Matrix with only translation decomposes correctly."""
    mat = np.eye(4, dtype=np.float32)
    mat[3, 0] = 1.0  # row-major transposed: translation in last row
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


def test_uniform_scale():
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


def test_round_trip():
    """Compose then decompose recovers original pos/quat/scale."""
    pos = np.array([[5.0, 6.0, 7.0]], dtype=np.float32)
    s45 = np.sin(np.pi / 4)
    c45 = np.cos(np.pi / 4)
    rot = np.array([[0.0, 0.0, s45, c45]], dtype=np.float32)
    scale = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

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

    np.testing.assert_allclose(out_pos.numpy()[0], pos[0], atol=1e-5)
    np.testing.assert_allclose(out_scale.numpy()[0], scale[0], atol=1e-5)
    r_out = out_rot.numpy()[0]
    r_exp = rot[0]
    dot = np.dot(r_out, r_exp)
    np.testing.assert_allclose(abs(dot), 1.0, atol=1e-5)


def test_non_uniform_scale_round_trip():
    """Non-uniform scale round-trips correctly."""
    pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    rot = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
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


# ------------------------------------------------------------------
# World ↔ Local matrix tests
#
# These test the same math the production fabric kernels use:
#   local^T = world^T * inv(parent^T)
#   world^T = local^T * parent^T
#
# Under the transposed storage convention, this is equivalent to:
#   local = inv(parent) * world
#   world = parent * local
# ------------------------------------------------------------------


def test_local_from_world_identity_parent():
    """With identity parent, local should equal world."""
    child_world_T = _make_transform_matrix([3, -1, 7], [0, 0, 0.3826834, 0.9238795], [1.5, 2.0, 0.5])
    parent_world_T = _make_transform_matrix([0, 0, 0], [0, 0, 0, 1], [1, 1, 1])

    cw = wp.array(child_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    pw = wp.array(parent_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    out = wp.zeros(1, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_local_from_world_kernel, dim=1, inputs=[cw, pw, out], device="cpu")
    wp.synchronize()

    np.testing.assert_allclose(out.numpy()[0], child_world_T, atol=1e-5)


def test_local_from_world_non_orthogonal():
    """Non-uniform scale + rotation in parent produces non-orthogonal local matrix.

    Verifies: local^T @ parent^T == child_world^T (reconstruction check).
    """
    parent_world_T = _make_transform_matrix([10, -5, 2], [0, 0.2588190, 0, 0.9659258], [2.0, 0.5, 3.0])
    child_world_T = _make_transform_matrix([1, 2, 3], [0.5, 0, 0, 0.8660254], [1.0, 1.5, 0.8])

    cw = wp.array(child_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    pw = wp.array(parent_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    out = wp.zeros(1, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_local_from_world_kernel, dim=1, inputs=[cw, pw, out], device="cpu")
    wp.synchronize()

    # Verify reconstruction: local^T @ parent^T should equal child_world^T
    local_T = out.numpy()[0]
    reconstructed = local_T @ parent_world_T
    np.testing.assert_allclose(reconstructed, child_world_T, atol=1e-5)


def test_world_from_local_non_orthogonal():
    """Recompose world from local with non-orthogonal parent."""
    parent_world_T = _make_transform_matrix([10, -5, 2], [0, 0.2588190, 0, 0.9659258], [2.0, 0.5, 3.0])
    child_world_T = _make_transform_matrix([1, 2, 3], [0.5, 0, 0, 0.8660254], [1.0, 1.5, 0.8])

    # Ground-truth local
    child_local_T = child_world_T @ np.linalg.inv(parent_world_T)

    cl = wp.array(child_local_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    pw = wp.array(parent_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    out = wp.zeros(1, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_world_from_local_kernel, dim=1, inputs=[cl, pw, out], device="cpu")
    wp.synchronize()

    np.testing.assert_allclose(out.numpy()[0], child_world_T, atol=1e-5)


def test_world_local_round_trip_non_orthogonal_batch():
    """Batch of 4 prims with non-orthogonal transforms: world->local->world round-trip."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(42)
    n = 4

    parent_scales = rng.uniform(0.3, 3.0, size=(n, 3)).astype(np.float64)
    child_scales = rng.uniform(0.3, 3.0, size=(n, 3)).astype(np.float64)
    parent_rots = Rotation.random(n, random_state=42).as_quat().astype(np.float64)
    child_rots = Rotation.random(n, random_state=99).as_quat().astype(np.float64)
    parent_positions = rng.uniform(-10, 10, size=(n, 3)).astype(np.float64)
    child_positions = rng.uniform(-10, 10, size=(n, 3)).astype(np.float64)

    parent_world_Ts = np.stack(
        [_make_transform_matrix(parent_positions[i], parent_rots[i], parent_scales[i]) for i in range(n)]
    )
    child_world_Ts = np.stack(
        [_make_transform_matrix(child_positions[i], child_rots[i], child_scales[i]) for i in range(n)]
    )

    # world -> local
    cw = wp.array(child_world_Ts, dtype=wp.mat44d, device="cpu")
    pw = wp.array(parent_world_Ts, dtype=wp.mat44d, device="cpu")
    local_out = wp.zeros(n, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_local_from_world_kernel, dim=n, inputs=[cw, pw, local_out], device="cpu")
    wp.synchronize()

    # local -> world (round-trip)
    cl = wp.array(local_out.numpy(), dtype=wp.mat44d, device="cpu")
    pw2 = wp.array(parent_world_Ts, dtype=wp.mat44d, device="cpu")
    world_out = wp.zeros(n, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_world_from_local_kernel, dim=n, inputs=[cl, pw2, world_out], device="cpu")
    wp.synchronize()

    np.testing.assert_allclose(world_out.numpy(), child_world_Ts, atol=1e-4)


def test_local_from_world_sheared_parent():
    """Parent with extreme non-uniform scale (10:1 ratio) creating significant shear.

    Verifies both correctness and that the resulting local matrix is genuinely
    non-orthogonal (the whole point of testing with sheared transforms).
    """
    parent_world_T = _make_transform_matrix([0, 0, 0], [0, 0, 0.3826834, 0.9238795], [10.0, 1.0, 1.0])
    child_world_T = _make_transform_matrix([5, 5, 0], [0, 0, 0, 1], [1.0, 1.0, 1.0])

    cw = wp.array(child_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    pw = wp.array(parent_world_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    out = wp.zeros(1, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_local_from_world_kernel, dim=1, inputs=[cw, pw, out], device="cpu")
    wp.synchronize()

    # Verify reconstruction
    local_T = out.numpy()[0]
    reconstructed = local_T @ parent_world_T
    np.testing.assert_allclose(reconstructed, child_world_T, atol=1e-4)

    # Verify the local matrix upper-3x3 is NOT orthogonal
    upper3x3 = local_T[:3, :3]
    gram = upper3x3 @ upper3x3.T
    assert not np.allclose(gram, np.eye(3), atol=0.1), (
        "Local matrix upper-3x3 should NOT be orthogonal with 10:1 sheared parent"
    )


# ------------------------------------------------------------------
# Kernel signature / importability tests
# ------------------------------------------------------------------


def test_all_kernels_importable():
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


def test_module_exports_match_all():
    """__all__ should list every public kernel."""
    from isaaclab.utils.warp import fabric as fabric_utils

    for name in fabric_utils.__all__:
        assert hasattr(fabric_utils, name), f"__all__ lists '{name}' but it's not defined"
