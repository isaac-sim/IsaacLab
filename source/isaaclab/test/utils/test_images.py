# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`isaaclab.utils.images`."""

from __future__ import annotations

import pytest
import torch
import warp as wp

wp.config.quiet = True
wp.init()


@pytest.fixture(params=["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return request.param


class TestPredicates:
    """The is_* dispatch predicates."""

    @pytest.mark.parametrize(
        "data_type",
        ["rgb", "rgba", "albedo", "simple_shading_constant_diffuse", "simple_shading_diffuse_mdl"],
    )
    def test_is_rgb_like_matches(self, data_type):
        from isaaclab.utils.images import is_rgb_like

        assert is_rgb_like(data_type)

    @pytest.mark.parametrize("data_type", ["depth", "distance_to_camera", "normals", "semantic_segmentation"])
    def test_is_rgb_like_rejects_non_rgb(self, data_type):
        from isaaclab.utils.images import is_rgb_like

        assert not is_rgb_like(data_type)

    @pytest.mark.parametrize("data_type", ["depth", "depth_linear", "distance_to_camera", "distance_to_plane"])
    def test_is_depth_like_matches(self, data_type):
        from isaaclab.utils.images import is_depth_like

        assert is_depth_like(data_type)

    @pytest.mark.parametrize("data_type", ["rgb", "albedo", "normals"])
    def test_is_depth_like_rejects(self, data_type):
        from isaaclab.utils.images import is_depth_like

        assert not is_depth_like(data_type)

    @pytest.mark.parametrize("data_type", ["normals", "normals_object_frame"])
    def test_is_normals_like_matches(self, data_type):
        from isaaclab.utils.images import is_normals_like

        assert is_normals_like(data_type)

    @pytest.mark.parametrize("data_type", ["rgb", "albedo", "depth", "distance_to_camera"])
    def test_is_normals_like_rejects(self, data_type):
        from isaaclab.utils.images import is_normals_like

        assert not is_normals_like(data_type)


class TestNormalizeCameraImageRGBLike:
    """RGB-like dispatch: rgb, albedo, simple_shading_*."""

    @pytest.mark.parametrize("data_type", ["rgb", "albedo", "simple_shading_diffuse_mdl"])
    def test_uint8_routes_to_warp_fast_path(self, device, data_type):
        """uint8 contiguous 4D input produces float32 ``(x/255 - per-image mean)`` output."""
        from isaaclab.utils.images import normalize_camera_image

        torch.manual_seed(0)
        src = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8, device=device)
        out = normalize_camera_image(src, data_type)

        expected = src.float() / 255.0
        expected = expected - torch.mean(expected, dim=(1, 2), keepdim=True)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        assert out.dtype == torch.float32

    def test_uint8_preallocated_output_reused(self, device):
        """``out`` kwarg is forwarded so callers can reuse storage."""
        from isaaclab.utils.images import normalize_camera_image

        src = torch.randint(0, 255, (2, 8, 8, 6), dtype=torch.uint8, device=device)
        out = torch.empty(src.shape, dtype=torch.float32, device=device)
        ptr = out.data_ptr()
        result = normalize_camera_image(src, "rgb", out=out)
        assert result is out
        assert result.data_ptr() == ptr

    def test_float_input_takes_pytorch_fallback(self, device):
        """Non-uint8 input routes through the PyTorch fallback with equivalent math."""
        from isaaclab.utils.images import normalize_camera_image

        torch.manual_seed(0)
        src_f = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8, device=device).float()
        out = normalize_camera_image(src_f, "rgb")

        expected = src_f / 255.0
        expected = expected - torch.mean(expected, dim=(1, 2), keepdim=True)
        torch.testing.assert_close(out, expected)
        assert out.dtype == torch.float32

    def test_non_contiguous_uint8_takes_pytorch_fallback(self, device):
        """Strided uint8 input falls back instead of raising in the Warp wrapper."""
        from isaaclab.utils.images import normalize_camera_image

        torch.manual_seed(0)
        base = torch.randint(0, 255, (2, 8, 8, 12), dtype=torch.uint8, device=device)
        src = base[..., ::2]
        assert not src.is_contiguous()
        out = normalize_camera_image(src, "rgb")

        ref = src.float() / 255.0
        expected = ref - torch.mean(ref, dim=(1, 2), keepdim=True)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_bchw_uint8_routes_to_warp_fast_path(self, device):
        """``channel_dim=1`` produces a BCHW-correct normalize via the Warp fast path."""
        from isaaclab.utils.images import normalize_camera_image

        torch.manual_seed(0)
        src = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8, device=device)
        out = normalize_camera_image(src, "rgb", channel_dim=1)

        expected = src.float() / 255.0
        expected = expected - torch.mean(expected, dim=(2, 3), keepdim=True)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        assert out.dtype == torch.float32

    def test_bchw_float_input_takes_pytorch_fallback(self, device):
        """Non-uint8 BCHW input routes through the PyTorch fallback with BCHW reduction axes."""
        from isaaclab.utils.images import normalize_camera_image

        torch.manual_seed(0)
        src_f = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8, device=device).float()
        out = normalize_camera_image(src_f, "rgb", channel_dim=1)

        expected = src_f / 255.0
        expected = expected - torch.mean(expected, dim=(2, 3), keepdim=True)
        torch.testing.assert_close(out, expected)


class TestNormalizeCameraImageDepth:
    """Depth-like dispatch: in-place ``inf -> 0``."""

    @pytest.mark.parametrize("data_type", ["depth", "distance_to_camera", "distance_to_plane"])
    def test_inf_replaced_with_zero_in_place(self, device, data_type):
        from isaaclab.utils.images import normalize_camera_image

        src = torch.tensor([[1.0, float("inf"), 3.0], [float("inf"), 2.0, 4.0]], device=device)
        out = normalize_camera_image(src, data_type)
        assert out is src
        expected = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]], device=device)
        torch.testing.assert_close(out, expected)


class TestNormalizeCameraImageNormals:
    """Normals dispatch: ``[-1, 1] -> [0, 1]``."""

    def test_range_remap(self, device):
        from isaaclab.utils.images import normalize_camera_image

        src = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
        out = normalize_camera_image(src, "normals")
        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
        torch.testing.assert_close(out, expected)


class TestNormalizeCameraImagePassthrough:
    """Unknown data_types return the input unchanged."""

    @pytest.mark.parametrize("data_type", ["semantic_segmentation", "instance_segmentation", "motion_vectors"])
    def test_unknown_type_passthrough(self, device, data_type):
        from isaaclab.utils.images import normalize_camera_image

        src = torch.ones((2, 4, 4, 3), device=device)
        out = normalize_camera_image(src, data_type)
        assert out is src
