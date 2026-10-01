# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the target-frame rebase logic, _to_numpy_4x4 helper, and config-driven auto-selection.

These tests need no Omniverse/Isaac Sim stack.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle, _to_numpy_4x4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# Rebase of the world_T_anchor external input
# ---------------------------------------------------------------------------


def _world_T_anchor_input(target_T_world) -> np.ndarray:
    """Run ``_build_external_inputs`` for a session that only consumes ``world_T_anchor``."""
    ValueInput = pytest.importorskip("isaacteleop.retargeting_engine.interface").ValueInput

    lifecycle = object.__new__(TeleopSessionLifecycle)
    lifecycle._session = MagicMock()
    lifecycle._session.has_external_inputs.return_value = True
    lifecycle._session.get_external_input_specs.return_value = [TeleopSessionLifecycle.WORLD_T_ANCHOR_INPUT_NAME]

    world_T_anchor = np.eye(4, dtype=np.float32)
    world_T_anchor[:3, 3] = [1.0, 0.0, 0.0]

    external_inputs = lifecycle._build_external_inputs(lambda: world_T_anchor, target_T_world)
    return np.asarray(external_inputs[TeleopSessionLifecycle.WORLD_T_ANCHOR_INPUT_NAME][ValueInput.VALUE][0])


class TestBuildExternalInputsRebase:
    def test_target_T_world_left_multiplies_the_anchor(self, rotation_90z_matrix: np.ndarray):
        """The anchor input becomes ``target_T_world @ world_T_anchor`` (order matters for rotation + translation)."""
        target_T_world = rotation_90z_matrix.copy()
        target_T_world[:3, 3] = [10.0, 0.0, 0.0]

        # Rotating the anchor offset (1, 0, 0) by 90 deg about Z gives (0, 1, 0), then the target translation applies.
        expected = np.array(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(_world_T_anchor_input(target_T_world), expected)

    def test_none_target_leaves_anchor_unchanged(self):
        expected = np.eye(4, dtype=np.float32)
        expected[:3, 3] = [1.0, 0.0, 0.0]
        np.testing.assert_array_equal(_world_T_anchor_input(None), expected)


# ---------------------------------------------------------------------------
# Config-driven auto-selection in IsaacTeleopDevice.advance()
# ---------------------------------------------------------------------------

_EXPLICIT = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)
_AUTO = np.diag([3.0, 3.0, 3.0, 1.0]).astype(np.float32)


@pytest.mark.parametrize(
    ("prim_path", "explicit", "auto_result", "expected", "expect_auto_read"),
    [
        pytest.param(None, None, _AUTO, None, False, id="no-config-no-explicit"),
        pytest.param(None, _EXPLICIT, _AUTO, _EXPLICIT, False, id="explicit-only"),
        pytest.param("/World/Robot/base_link", None, _AUTO, _AUTO, True, id="config-auto-read"),
        pytest.param("/World/Robot/base_link", _EXPLICIT, _AUTO, _EXPLICIT, False, id="explicit-overrides-config"),
        pytest.param("/World/Robot/base_link", None, None, None, True, id="config-read-fails"),
    ],
)
def test_advance_selects_target_frame(prim_path, explicit, auto_result, expected, expect_auto_read):
    """advance() passes an explicit target_T_world through, else reads the configured prim."""
    device = object.__new__(IsaacTeleopDevice)
    device._cfg = SimpleNamespace(target_frame_prim_path=prim_path)
    device._session_lifecycle = MagicMock()
    device._session_lifecycle.step.return_value = None
    device._anchor_manager = MagicMock()
    device._get_target_frame_T_world = MagicMock(return_value=auto_result)
    device._dispatch_control_callbacks = lambda: None

    device.advance(target_T_world=explicit)

    assert device._get_target_frame_T_world.called is expect_auto_read
    step_kwargs = device._session_lifecycle.step.call_args.kwargs
    assert step_kwargs["anchor_world_matrix_fn"] is device._anchor_manager.get_world_matrix
    assert step_kwargs["target_T_world"] is expected
