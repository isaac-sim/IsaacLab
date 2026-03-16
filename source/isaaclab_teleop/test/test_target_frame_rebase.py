# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the target-frame rebase logic and _to_numpy_4x4 helper.

These tests exercise pure math (no Omniverse/Isaac Sim stack required).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isaaclab_teleop.session_lifecycle import _to_numpy_4x4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_4x4() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


@pytest.fixture
def translation_matrix() -> np.ndarray:
    """A pure translation of (1, 2, 3)."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = [1.0, 2.0, 3.0]
    return mat


@pytest.fixture
def rotation_90z_matrix() -> np.ndarray:
    """90-degree rotation about Z axis."""
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = 0.0
    mat[0, 1] = -1.0
    mat[1, 0] = 1.0
    mat[1, 1] = 0.0
    return mat


# ---------------------------------------------------------------------------
# _to_numpy_4x4 conversion tests
# ---------------------------------------------------------------------------


class TestToNumpy4x4:
    def test_from_ndarray_float32(self, identity_4x4: np.ndarray):
        result = _to_numpy_4x4(identity_4x4)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, identity_4x4)

    def test_from_ndarray_float64_casts(self):
        mat = np.eye(4, dtype=np.float64)
        result = _to_numpy_4x4(mat)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, np.eye(4, dtype=np.float32))

    def test_from_torch_cpu(self, translation_matrix: np.ndarray):
        tensor = torch.from_numpy(translation_matrix.copy())
        result = _to_numpy_4x4(tensor)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, translation_matrix)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_from_torch_gpu(self, translation_matrix: np.ndarray):
        tensor = torch.from_numpy(translation_matrix.copy()).cuda()
        result = _to_numpy_4x4(tensor)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, translation_matrix)

    def test_from_duck_typed_numpy(self, rotation_90z_matrix: np.ndarray):
        """Simulates a wp.array or similar object with a .numpy() method."""

        class FakeWarpArray:
            def __init__(self, data: np.ndarray):
                self._data = data

            def numpy(self) -> np.ndarray:
                return self._data

        fake = FakeWarpArray(rotation_90z_matrix.copy())
        result = _to_numpy_4x4(fake)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, rotation_90z_matrix)

    def test_from_list(self):
        data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _to_numpy_4x4(data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.eye(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# Matrix multiplication (rebase) tests
# ---------------------------------------------------------------------------


class TestRebaseMultiplication:
    def test_rebase_identity_is_noop(self, translation_matrix: np.ndarray):
        """target_T_world = I should leave anchor_matrix unchanged."""
        identity = np.eye(4, dtype=np.float32)
        result = _to_numpy_4x4(identity) @ translation_matrix
        np.testing.assert_array_almost_equal(result, translation_matrix)

    def test_rebase_translation(self, identity_4x4: np.ndarray):
        """Rebasing by a pure translation offsets the origin."""
        target_T_world = np.eye(4, dtype=np.float32)
        target_T_world[:3, 3] = [10.0, 20.0, 30.0]

        world_T_anchor = np.eye(4, dtype=np.float32)
        world_T_anchor[:3, 3] = [1.0, 2.0, 3.0]

        result = _to_numpy_4x4(target_T_world) @ world_T_anchor
        np.testing.assert_array_almost_equal(result[:3, 3], [11.0, 22.0, 33.0])

    def test_rebase_rotation(self, rotation_90z_matrix: np.ndarray):
        """A 90-deg Z rotation rebase should rotate the anchor translation."""
        world_T_anchor = np.eye(4, dtype=np.float32)
        world_T_anchor[:3, 3] = [1.0, 0.0, 0.0]

        result = _to_numpy_4x4(rotation_90z_matrix) @ world_T_anchor

        # After 90-deg Z rotation: (1,0,0) -> (0,1,0)
        np.testing.assert_array_almost_equal(result[:3, 3], [0.0, 1.0, 0.0])
        # Rotation part should match the 90-deg Z rotation
        np.testing.assert_array_almost_equal(result[:3, :3], rotation_90z_matrix[:3, :3])

    def test_rebase_with_torch_tensor(self, translation_matrix: np.ndarray):
        """target_T_world as a torch.Tensor should work identically."""
        target_T_world = torch.eye(4, dtype=torch.float32)
        target_T_world[:3, 3] = torch.tensor([5.0, 5.0, 5.0])

        result = _to_numpy_4x4(target_T_world) @ translation_matrix

        expected = np.eye(4, dtype=np.float32)
        expected[:3, 3] = [6.0, 7.0, 8.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_none_target_leaves_anchor_unchanged(self, translation_matrix: np.ndarray):
        """When target_T_world is None, the calling code should skip multiplication."""
        target_T_world = None
        anchor_matrix = translation_matrix.copy()

        if target_T_world is not None:
            anchor_matrix = _to_numpy_4x4(target_T_world) @ anchor_matrix

        np.testing.assert_array_equal(anchor_matrix, translation_matrix)

    def test_inverse_rebase_recovers_identity(self):
        """target_T_world = inv(world_T_anchor) should yield identity."""
        world_T_anchor = np.array(
            [
                [0.0, -1.0, 0.0, 3.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target_T_world = np.linalg.inv(world_T_anchor).astype(np.float32)

        result = _to_numpy_4x4(target_T_world) @ world_T_anchor
        np.testing.assert_array_almost_equal(result, np.eye(4, dtype=np.float32), decimal=5)
