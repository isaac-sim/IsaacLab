# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the target-frame rebase logic, _to_numpy_4x4 helper, and config-driven auto-selection.

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


# ---------------------------------------------------------------------------
# Config-driven auto-selection tests
# ---------------------------------------------------------------------------


def _simulate_advance_selection(
    target_T_world: np.ndarray | None,
    target_frame_prim_path: str | None,
    auto_read_result: np.ndarray | None = None,
) -> tuple[np.ndarray | None, bool]:
    """Replicate the auto-selection logic from IsaacTeleopDevice.advance().

    Returns the target_T_world that would be passed to step(), and whether
    _get_target_frame_T_world would have been called.
    """
    auto_read_called = False

    def fake_get_target_frame_T_world():
        nonlocal auto_read_called
        auto_read_called = True
        return auto_read_result

    if target_T_world is None and target_frame_prim_path is not None:
        target_T_world = fake_get_target_frame_T_world()

    return target_T_world, auto_read_called


class TestConfigDrivenAutoSelection:
    """Tests for the advance() auto-selection logic between explicit target_T_world
    and config-driven target_frame_prim_path.

    These tests replicate the branching logic from advance() without importing
    the full IsaacTeleopDevice (which requires Isaac Sim runtime dependencies).
    """

    def test_no_config_no_explicit_passes_none(self):
        """When neither config nor explicit target_T_world is set, step() receives None."""
        result, called = _simulate_advance_selection(target_T_world=None, target_frame_prim_path=None)
        assert result is None
        assert not called

    def test_explicit_target_is_passed_through(self):
        """An explicit target_T_world should be passed directly to step()."""
        explicit = np.eye(4, dtype=np.float32)
        explicit[:3, 3] = [1.0, 2.0, 3.0]

        result, called = _simulate_advance_selection(target_T_world=explicit, target_frame_prim_path=None)
        np.testing.assert_array_equal(result, explicit)
        assert not called

    def test_config_prim_triggers_auto_read(self):
        """When target_frame_prim_path is set, _get_target_frame_T_world is called."""
        auto_matrix = np.eye(4, dtype=np.float32)
        auto_matrix[:3, 3] = [9.0, 8.0, 7.0]

        result, called = _simulate_advance_selection(
            target_T_world=None,
            target_frame_prim_path="/World/Robot/base_link",
            auto_read_result=auto_matrix,
        )
        assert called
        np.testing.assert_array_equal(result, auto_matrix)

    def test_explicit_overrides_config(self):
        """An explicit target_T_world takes precedence over config prim path."""
        explicit = np.eye(4, dtype=np.float32)
        explicit[:3, 3] = [42.0, 0.0, 0.0]

        result, called = _simulate_advance_selection(
            target_T_world=explicit,
            target_frame_prim_path="/World/Robot/base_link",
            auto_read_result=np.eye(4, dtype=np.float32),
        )
        assert not called
        np.testing.assert_array_equal(result, explicit)

    def test_config_prim_returns_none_passes_none(self):
        """If the prim read fails (returns None), step() receives None."""
        result, called = _simulate_advance_selection(
            target_T_world=None,
            target_frame_prim_path="/World/Robot/base_link",
            auto_read_result=None,
        )
        assert called
        assert result is None
