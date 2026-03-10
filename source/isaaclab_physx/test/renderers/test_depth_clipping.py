# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for apply_depth_clipping and DEPTH_DATA_TYPES."""

import pytest
import torch

from isaaclab_physx.renderers.isaac_rtx_renderer_utils import DEPTH_DATA_TYPES, apply_depth_clipping

CLIPPING_RANGE = (0.1, 100.0)
FAR = CLIPPING_RANGE[1]


def _make_depth_tensor(*values: float, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# DEPTH_DATA_TYPES constant
# ---------------------------------------------------------------------------


def test_depth_data_types_contains_expected_entries():
    assert "distance_to_camera" in DEPTH_DATA_TYPES
    assert "distance_to_image_plane" in DEPTH_DATA_TYPES
    assert "depth" in DEPTH_DATA_TYPES


def test_depth_data_types_excludes_non_depth():
    assert "rgb" not in DEPTH_DATA_TYPES
    assert "rgba" not in DEPTH_DATA_TYPES
    assert "normals" not in DEPTH_DATA_TYPES


# ---------------------------------------------------------------------------
# distance_to_camera: radial-clipping step
# ---------------------------------------------------------------------------


def test_distance_to_camera_exceeding_far_becomes_inf():
    """Values > far must become inf before the behavior step."""
    t = _make_depth_tensor(50.0, 100.1, 200.0, float("inf"))
    apply_depth_clipping(t, "distance_to_camera", CLIPPING_RANGE, "none")
    assert t[0].item() == 50.0
    assert torch.isinf(t[1])
    assert torch.isinf(t[2])
    assert torch.isinf(t[3])


def test_distance_to_camera_at_far_boundary_unchanged():
    """Exactly-at-far should NOT be clipped (only strictly greater)."""
    t = _make_depth_tensor(FAR)
    apply_depth_clipping(t, "distance_to_camera", CLIPPING_RANGE, "none")
    assert t[0].item() == FAR


# ---------------------------------------------------------------------------
# behavior="zero"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_type", sorted(DEPTH_DATA_TYPES))
def test_behavior_zero_replaces_inf_with_zero(data_type):
    t = _make_depth_tensor(10.0, float("inf"), 50.0)
    apply_depth_clipping(t, data_type, CLIPPING_RANGE, "zero")
    assert t[0].item() == 10.0
    assert t[1].item() == 0.0
    assert t[2].item() == 50.0


def test_distance_to_camera_zero_clips_and_zeros():
    """Exceeding-far values should first become inf, then be zeroed."""
    t = _make_depth_tensor(50.0, 150.0)
    apply_depth_clipping(t, "distance_to_camera", CLIPPING_RANGE, "zero")
    assert t[0].item() == 50.0
    assert t[1].item() == 0.0


# ---------------------------------------------------------------------------
# behavior="max"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_type", sorted(DEPTH_DATA_TYPES))
def test_behavior_max_replaces_inf_with_far(data_type):
    t = _make_depth_tensor(10.0, float("inf"), 50.0)
    apply_depth_clipping(t, data_type, CLIPPING_RANGE, "max")
    assert t[0].item() == 10.0
    assert t[1].item() == FAR
    assert t[2].item() == 50.0


def test_distance_to_camera_max_clips_and_maxes():
    """Exceeding-far values should first become inf, then be set to far."""
    t = _make_depth_tensor(50.0, 150.0)
    apply_depth_clipping(t, "distance_to_camera", CLIPPING_RANGE, "max")
    assert t[0].item() == 50.0
    assert t[1].item() == FAR


# ---------------------------------------------------------------------------
# behavior="none"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_type", ["distance_to_image_plane", "depth"])
def test_behavior_none_preserves_inf(data_type):
    t = _make_depth_tensor(10.0, float("inf"))
    apply_depth_clipping(t, data_type, CLIPPING_RANGE, "none")
    assert t[0].item() == 10.0
    assert torch.isinf(t[1])


# ---------------------------------------------------------------------------
# Non-depth data types are unaffected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_type", ["rgb", "rgba", "normals", "semantic_segmentation"])
def test_non_depth_data_type_unchanged(data_type):
    original = _make_depth_tensor(10.0, 200.0, float("inf"))
    t = original.clone()
    apply_depth_clipping(t, data_type, CLIPPING_RANGE, "max")
    assert torch.equal(t, original)


# ---------------------------------------------------------------------------
# In-place mutation
# ---------------------------------------------------------------------------


def test_modifies_tensor_in_place():
    t = _make_depth_tensor(float("inf"))
    original_data_ptr = t.data_ptr()
    apply_depth_clipping(t, "depth", CLIPPING_RANGE, "zero")
    assert t.data_ptr() == original_data_ptr
    assert t[0].item() == 0.0


# ---------------------------------------------------------------------------
# Multi-dimensional tensors (realistic camera output shapes)
# ---------------------------------------------------------------------------


def test_works_on_batched_image_tensor():
    """Typical shape: (num_envs, H, W)."""
    t = torch.tensor(
        [
            [[50.0, 150.0], [float("inf"), 80.0]],
            [[200.0, 10.0], [30.0, float("inf")]],
        ],
        dtype=torch.float32,
    )
    apply_depth_clipping(t, "distance_to_camera", CLIPPING_RANGE, "zero")
    assert t[0, 0, 0].item() == 50.0
    assert t[0, 0, 1].item() == 0.0
    assert t[0, 1, 0].item() == 0.0
    assert t[0, 1, 1].item() == 80.0
    assert t[1, 0, 0].item() == 0.0
    assert t[1, 0, 1].item() == 10.0
    assert t[1, 1, 0].item() == 30.0
    assert t[1, 1, 1].item() == 0.0


# ---------------------------------------------------------------------------
# CUDA device (if available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor():
    t = _make_depth_tensor(50.0, 150.0, float("inf"), device="cuda:0")
    apply_depth_clipping(t, "distance_to_camera", CLIPPING_RANGE, "max")
    assert t[0].item() == 50.0
    assert t[1].item() == FAR
    assert t[2].item() == FAR
