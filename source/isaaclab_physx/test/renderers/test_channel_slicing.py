# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for channel slicing rules and SEGMENTATION_COLORIZE_FIELDS."""

import pytest
import torch
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
    ANNOTATOR_CHANNEL_COUNTS,
    SEGMENTATION_COLORIZE_FIELDS,
    SIMPLE_SHADING_MODES,
    slice_output_channels,
)

# ---------------------------------------------------------------------------
# ANNOTATOR_CHANNEL_COUNTS constant
# ---------------------------------------------------------------------------


class TestAnnotatorChannelCounts:
    def test_motion_vectors_has_two_channels(self):
        assert ANNOTATOR_CHANNEL_COUNTS["motion_vectors"] == 2

    def test_normals_has_three_channels(self):
        assert ANNOTATOR_CHANNEL_COUNTS["normals"] == 3

    def test_rgb_has_three_channels(self):
        assert ANNOTATOR_CHANNEL_COUNTS["rgb"] == 3

    @pytest.mark.parametrize("mode", sorted(SIMPLE_SHADING_MODES))
    def test_simple_shading_modes_have_three_channels(self, mode):
        assert ANNOTATOR_CHANNEL_COUNTS[mode] == 3

    def test_does_not_contain_depth_types(self):
        for name in ("depth", "distance_to_camera", "distance_to_image_plane"):
            assert name not in ANNOTATOR_CHANNEL_COUNTS

    def test_does_not_contain_segmentation_types(self):
        for name in ("semantic_segmentation", "instance_segmentation_fast", "instance_id_segmentation_fast"):
            assert name not in ANNOTATOR_CHANNEL_COUNTS

    def test_does_not_contain_rgba(self):
        assert "rgba" not in ANNOTATOR_CHANNEL_COUNTS


# ---------------------------------------------------------------------------
# SEGMENTATION_COLORIZE_FIELDS constant
# ---------------------------------------------------------------------------


class TestSegmentationColorizeFields:
    @pytest.mark.parametrize(
        "data_type,expected_field",
        [
            ("semantic_segmentation", "colorize_semantic_segmentation"),
            ("instance_segmentation_fast", "colorize_instance_segmentation"),
            ("instance_id_segmentation_fast", "colorize_instance_id_segmentation"),
        ],
    )
    def test_maps_to_expected_config_field(self, data_type, expected_field):
        assert SEGMENTATION_COLORIZE_FIELDS[data_type] == expected_field

    def test_non_segmentation_types_absent(self):
        for name in ("rgb", "depth", "normals", "motion_vectors"):
            assert name not in SEGMENTATION_COLORIZE_FIELDS


# ---------------------------------------------------------------------------
# slice_output_channels
# ---------------------------------------------------------------------------


class TestSliceOutputChannels:
    @pytest.mark.parametrize(
        "data_type,expected_channels",
        [
            ("motion_vectors", 2),
            ("normals", 3),
            ("rgb", 3),
        ]
        + [(mode, 3) for mode in sorted(SIMPLE_SHADING_MODES)],
    )
    def test_slices_to_expected_channels(self, data_type, expected_channels):
        data = torch.rand(8, 8, 4)
        result = slice_output_channels(data, data_type)
        assert result.shape == (8, 8, expected_channels)

    @pytest.mark.parametrize("data_type", ["rgba", "depth", "distance_to_camera", "semantic_segmentation", "unknown"])
    def test_returns_unchanged_for_unlisted_types(self, data_type):
        data = torch.rand(8, 8, 4)
        result = slice_output_channels(data, data_type)
        assert result is data

    def test_returns_view_not_copy(self):
        data = torch.rand(8, 8, 4)
        result = slice_output_channels(data, "normals")
        assert result.data_ptr() == data.data_ptr()
        assert result.shape[-1] == 3

    def test_works_on_batched_images(self):
        data = torch.rand(4, 16, 16, 4)
        result = slice_output_channels(data, "motion_vectors")
        assert result.shape == (4, 16, 16, 2)

    def test_works_on_2d_tensor(self):
        data = torch.rand(10, 4)
        result = slice_output_channels(data, "rgb")
        assert result.shape == (10, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_works_on_cuda_tensor(self):
        data = torch.rand(8, 8, 4, device="cuda:0")
        result = slice_output_channels(data, "normals")
        assert result.shape == (8, 8, 3)
        assert result.device == data.device

    def test_sliced_values_match_source(self):
        data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        result = slice_output_channels(data, "motion_vectors")
        assert torch.equal(result, data[..., :2])

    def test_sliced_values_match_source_three_channels(self):
        data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        result = slice_output_channels(data, "normals")
        assert torch.equal(result, data[..., :3])
