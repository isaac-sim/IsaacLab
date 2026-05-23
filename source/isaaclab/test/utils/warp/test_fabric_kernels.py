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

from isaaclab.utils.warp.fabric import (  # noqa: E402
    _decompose_transformation_matrix,
    _local_from_world_transposed,
    _world_from_local_transposed,
)

# ------------------------------------------------------------------
# Test kernels — thin wp.array wrappers that delegate to production @wp.func
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _test_decompose_kernel(
    matrices: wp.array(dtype=wp.mat44f),
    out_positions: wp.array(dtype=wp.vec3f),
    out_rotations: wp.array(dtype=wp.quatf),
    out_scales: wp.array(dtype=wp.vec3f),
):
    """Decompose a batch of 4x4 matrices into pos/quat/scale."""
    i = wp.tid()
    pos, rot, scale = _decompose_transformation_matrix(matrices[i])
    out_positions[i] = pos
    out_rotations[i] = rot
    out_scales[i] = scale


@wp.kernel(enable_backward=False)
def _test_local_from_world_kernel(
    child_world: wp.array(dtype=wp.mat44d),
    parent_world: wp.array(dtype=wp.mat44d),
    out_local: wp.array(dtype=wp.mat44d),
):
    """wp.array adapter for _local_from_world_transposed — same func as production fabric kernel."""
    i = wp.tid()
    out_local[i] = wp.mat44d(_local_from_world_transposed(wp.mat44f(child_world[i]), wp.mat44f(parent_world[i])))


@wp.kernel(enable_backward=False)
def _test_world_from_local_kernel(
    child_local: wp.array(dtype=wp.mat44d),
    parent_world: wp.array(dtype=wp.mat44d),
    out_world: wp.array(dtype=wp.mat44d),
):
    """wp.array adapter for _world_from_local_transposed — same func as production fabric kernel."""
    i = wp.tid()
    out_world[i] = wp.mat44d(_world_from_local_transposed(wp.mat44f(child_local[i]), wp.mat44f(parent_world[i])))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_transform_matrix(pos, rot_quat_xyzw, scale):
    """Build a 4x4 Fabric-transposed transform from pos/quat/scale.

    Returns numpy (4,4) float64 in the transposed storage convention (row-major with
    translation in the last row) that Fabric uses.

    Raises:
        AssertionError: If the resulting matrix is singular (e.g. zero scale component).
    """
    p = wp.vec3f(*pos)
    q = wp.quatf(*rot_quat_xyzw)
    s = wp.vec3f(*scale)
    m = wp.transpose(wp.transform_compose(p, q, s))
    result = np.array(m).reshape(4, 4).astype(np.float64)
    det = np.linalg.det(result)
    assert abs(det) > 1e-6, f"Singular matrix: det={det:.2e}, scale={scale}"
    return result


# ------------------------------------------------------------------
# Decompose / Compose round-trip tests
# ------------------------------------------------------------------


def test_decompose_round_trip():
    """Decompose a matrix with translation, rotation, and non-uniform scale; verify round-trip."""
    pos = np.array([5.0, -3.0, 7.0])
    s45 = np.sin(np.pi / 4)
    c45 = np.cos(np.pi / 4)
    quat_xyzw = np.array([0.0, 0.0, s45, c45])  # 45° Z rotation
    scale = np.array([1.5, 0.8, 3.0])

    mat = _make_transform_matrix(pos, quat_xyzw, scale).astype(np.float32)
    matrices = wp.array(mat.reshape(1, 4, 4), dtype=wp.mat44f, device="cpu")

    out_pos = wp.zeros(1, dtype=wp.vec3f, device="cpu")
    out_rot = wp.zeros(1, dtype=wp.quatf, device="cpu")
    out_scale = wp.zeros(1, dtype=wp.vec3f, device="cpu")

    wp.launch(_test_decompose_kernel, dim=1, inputs=[matrices, out_pos, out_rot, out_scale], device="cpu")
    wp.synchronize()

    np.testing.assert_allclose(out_pos.numpy()[0], pos, atol=1e-5)
    np.testing.assert_allclose(out_scale.numpy()[0], scale, atol=1e-5)
    dot = np.dot(out_rot.numpy()[0], quat_xyzw)
    np.testing.assert_allclose(abs(dot), 1.0, atol=1e-5)


# ------------------------------------------------------------------
# World ↔ Local matrix tests
#
# These test the same math the production fabric kernels use:
#   local^T = world^T * inv(parent^T)
#   world^T = local^T * parent^T
#
# Both parent and child have rotation, translation, and non-uniform scale
# (producing sheared/non-orthogonal upper-3x3 blocks).
# ------------------------------------------------------------------

# Shared test data: parent with 10:1 non-uniform scale + 45° Z rotation + translation
_PARENT_WORLD_T = _make_transform_matrix([10, -5, 2], [0, 0, 0.3826834, 0.9238795], [4.0, 0.5, 2.0])
_CHILD_WORLD_T = _make_transform_matrix([1, 2, 3], [0.2588190, 0, 0, 0.9659258], [1.5, 0.8, 3.0])


def test_local_from_world_transposed():
    """local^T = world^T * inv(parent^T) — verified by reconstruction."""
    cw = wp.array(_CHILD_WORLD_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    pw = wp.array(_PARENT_WORLD_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    out = wp.zeros(1, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_local_from_world_kernel, dim=1, inputs=[cw, pw, out], device="cpu")
    wp.synchronize()

    # Reconstruction: local^T @ parent^T must equal child_world^T
    local_T = out.numpy()[0]
    reconstructed = local_T @ _PARENT_WORLD_T
    np.testing.assert_allclose(reconstructed, _CHILD_WORLD_T, atol=1e-5)


def test_world_from_local_transposed():
    """world^T = local^T * parent^T — verified against known child world."""
    # Ground-truth local computed via numpy
    child_local_T = _CHILD_WORLD_T @ np.linalg.inv(_PARENT_WORLD_T)

    cl = wp.array(child_local_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    pw = wp.array(_PARENT_WORLD_T.reshape(1, 4, 4), dtype=wp.mat44d, device="cpu")
    out = wp.zeros(1, dtype=wp.mat44d, device="cpu")

    wp.launch(_test_world_from_local_kernel, dim=1, inputs=[cl, pw, out], device="cpu")
    wp.synchronize()

    np.testing.assert_allclose(out.numpy()[0], _CHILD_WORLD_T, atol=1e-5)
