# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX RenderData (buffer creation, shapes)."""

import pytest

# Skip entire module if ovrtx not available (ovrtx_renderer imports it)
pytest.importorskip("ovrtx")

# Warp/GPU tests may need app context
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import warp as wp

from isaaclab.renderer.ovrtx_renderer import OVRTXRenderData


@pytest.fixture
def device():
    """Use CUDA if available, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


class TestOVRTXRenderData:
    """Tests for OVRTXRenderData buffer creation."""

    def test_rgb_buffers(self, device):
        """Test RGB/RGBA buffer creation and shapes."""
        rd = OVRTXRenderData(
            width=64,
            height=64,
            num_envs=4,
            data_types=["rgb"],
            device=device,
        )
        assert "rgba" in rd.warp_buffers
        assert "rgb" in rd.warp_buffers
        assert rd.warp_buffers["rgba"].shape == (4, 64, 64, 4)
        assert rd.warp_buffers["rgb"].shape == (4, 64, 64, 3)
        assert rd.width == 64
        assert rd.height == 64
        assert rd.num_envs == 4

    def test_depth_buffers(self, device):
        """Test depth buffer creation."""
        rd = OVRTXRenderData(
            width=100,
            height=100,
            num_envs=8,
            data_types=["depth"],
            device=device,
        )
        assert "depth" in rd.warp_buffers
        assert rd.warp_buffers["depth"].shape == (8, 100, 100, 1)
        assert rd.warp_buffers["depth"].dtype == wp.float32

    def test_multiple_data_types(self, device):
        """Test creation with multiple data types."""
        rd = OVRTXRenderData(
            width=128,
            height=128,
            num_envs=2,
            data_types=["rgb", "depth", "albedo", "semantic_segmentation"],
            device=device,
        )
        assert "rgba" in rd.warp_buffers
        assert "rgb" in rd.warp_buffers
        assert "depth" in rd.warp_buffers
        assert "albedo" in rd.warp_buffers
        assert "semantic_segmentation" in rd.warp_buffers

        for key in ["rgba", "albedo", "semantic_segmentation"]:
            assert rd.warp_buffers[key].shape == (2, 128, 128, 4)
            assert rd.warp_buffers[key].dtype == wp.uint8

    def test_depth_aliases(self, device):
        """Test distance_to_image_plane and distance_to_camera buffers."""
        rd = OVRTXRenderData(
            width=64,
            height=64,
            num_envs=1,
            data_types=["distance_to_image_plane", "distance_to_camera"],
            device=device,
        )
        assert "distance_to_image_plane" in rd.warp_buffers
        assert "distance_to_camera" in rd.warp_buffers
        for key in ["distance_to_image_plane", "distance_to_camera"]:
            assert rd.warp_buffers[key].shape == (1, 64, 64, 1)
            assert rd.warp_buffers[key].dtype == wp.float32

    def test_rgb_is_view_of_rgba(self, device):
        """Test that rgb buffer is a view of rgba (shared memory)."""
        rd = OVRTXRenderData(
            width=32,
            height=32,
            num_envs=1,
            data_types=["rgb"],
            device=device,
        )
        # rgb should be rgba[:,:,:,:3] - same underlying storage
        rgba = rd.warp_buffers["rgba"]
        rgb = rd.warp_buffers["rgb"]
        assert rgb.ptr == rgba.ptr
        assert rgb.shape == (1, 32, 32, 3)
