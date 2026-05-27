# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`isaaclab.utils.warp.ops.normalize_image_uint8`."""

from __future__ import annotations

import pytest
import torch
import warp as wp

wp.config.quiet = True
wp.init()


def _pytorch_reference(src: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    """Reference normalize: ``(x / 255 - per-image-channel mean)`` in pure PyTorch."""
    x = src.float() / 255.0
    resolved = channel_dim + x.ndim if channel_dim < 0 else channel_dim
    spatial_dims = tuple(d for d in range(1, x.ndim) if d != resolved)
    return x - torch.mean(x, dim=spatial_dims, keepdim=True)


@pytest.fixture(params=["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    """Parametrize across CPU and CUDA."""
    return request.param


class TestNormalizeImageUint8:
    """Tests for the Warp-backed fused uint8 normalize wrapper."""

    def test_matches_pytorch_reference_constant_input(self, device):
        """A constant-valued uint8 input must normalize to all zeros (mean equals every pixel)."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.full((2, 4, 4, 6), 128, dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_matches_pytorch_reference_random_input(self, device):
        """Output must match the pure-PyTorch reference on randomized input."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        torch.manual_seed(0)
        src = torch.randint(0, 255, (3, 16, 16, 6), dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src)
        expected = _pytorch_reference(src)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_matches_pytorch_reference_disjoint_channel_slices(self, device):
        """Two frames concatenated along C must each normalize independently per-channel-slice."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        torch.manual_seed(1)
        c = 3
        f1 = torch.randint(0, 255, (2, 8, 8, c), dtype=torch.uint8, device=device)
        f2 = torch.randint(0, 255, (2, 8, 8, c), dtype=torch.uint8, device=device)
        stacked = torch.cat([f1, f2], dim=-1).contiguous()
        out = normalize_image_uint8(stacked)
        torch.testing.assert_close(out[..., :c], _pytorch_reference(f1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[..., c:], _pytorch_reference(f2), atol=1e-5, rtol=1e-5)

    def test_preallocated_output_reused(self, device):
        """When ``out`` is passed in, the wrapper writes into it and returns the same object."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.randint(0, 255, (2, 8, 8, 6), dtype=torch.uint8, device=device)
        out = torch.empty(src.shape, dtype=torch.float32, device=device)
        ptr_before = out.data_ptr()
        result = normalize_image_uint8(src, out=out)
        assert result is out
        assert result.data_ptr() == ptr_before

        src2 = torch.randint(0, 255, (2, 8, 8, 6), dtype=torch.uint8, device=device)
        result2 = normalize_image_uint8(src2, out=out)
        assert result2 is out
        assert result2.data_ptr() == ptr_before
        torch.testing.assert_close(result2, _pytorch_reference(src2), atol=1e-5, rtol=1e-5)

    def test_rejects_non_uint8_input(self, device):
        """Float input is a programming error and must raise."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.zeros((2, 4, 4, 3), dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match="4D uint8"):
            normalize_image_uint8(src)

    def test_rejects_wrong_ndim(self, device):
        """3D uint8 input is rejected (kernel expects (B, H, W, C))."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.zeros((4, 4, 3), dtype=torch.uint8, device=device)
        with pytest.raises(ValueError, match="4D uint8"):
            normalize_image_uint8(src)

    def test_rejects_non_contiguous_input(self, device):
        """Non-contiguous src is rejected."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        base = torch.randint(0, 255, (2, 8, 8, 12), dtype=torch.uint8, device=device)
        src = base[..., ::2]
        assert not src.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            normalize_image_uint8(src)

    def test_rejects_out_shape_mismatch(self, device):
        """A pre-allocated ``out`` of the wrong shape must raise."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.zeros((2, 4, 4, 6), dtype=torch.uint8, device=device)
        bad_out = torch.empty((2, 4, 4, 3), dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match="out shape/dtype/device"):
            normalize_image_uint8(src, out=bad_out)

    def test_rejects_out_dtype_mismatch(self, device):
        """A pre-allocated ``out`` of the wrong dtype must raise."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.zeros((2, 4, 4, 6), dtype=torch.uint8, device=device)
        bad_out = torch.empty(src.shape, dtype=torch.float16, device=device)
        with pytest.raises(ValueError, match="out shape/dtype/device"):
            normalize_image_uint8(src, out=bad_out)

    def test_bchw_matches_pytorch_reference(self, device):
        """``channel_dim=1`` (BCHW) must match a BCHW-layout PyTorch reference."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        torch.manual_seed(2)
        src = torch.randint(0, 255, (3, 6, 16, 16), dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src, channel_dim=1)
        torch.testing.assert_close(out, _pytorch_reference(src, channel_dim=1), atol=1e-5, rtol=1e-5)

    def test_bchw_negative_index_equivalent_to_positive(self, device):
        """``channel_dim=-3`` must produce the same output as ``channel_dim=1`` for 4D input."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        torch.manual_seed(3)
        src = torch.randint(0, 255, (2, 4, 8, 8), dtype=torch.uint8, device=device)
        out_pos = normalize_image_uint8(src, channel_dim=1)
        out_neg = normalize_image_uint8(src, channel_dim=-3)
        torch.testing.assert_close(out_pos, out_neg)

    def test_bhwc_explicit_positive_index_matches_default(self, device):
        """``channel_dim=3`` and the default ``channel_dim=-1`` must agree on BHWC input."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        torch.manual_seed(4)
        src = torch.randint(0, 255, (2, 8, 8, 4), dtype=torch.uint8, device=device)
        out_default = normalize_image_uint8(src)
        out_explicit = normalize_image_uint8(src, channel_dim=3)
        torch.testing.assert_close(out_default, out_explicit)

    def test_bchw_disjoint_channel_slices(self, device):
        """K frames concatenated along C in BCHW must each normalize independently per-channel."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        torch.manual_seed(5)
        c = 3
        f1 = torch.randint(0, 255, (2, c, 8, 8), dtype=torch.uint8, device=device)
        f2 = torch.randint(0, 255, (2, c, 8, 8), dtype=torch.uint8, device=device)
        stacked = torch.cat([f1, f2], dim=1).contiguous()
        out = normalize_image_uint8(stacked, channel_dim=1)
        torch.testing.assert_close(out[:, :c], _pytorch_reference(f1, channel_dim=1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[:, c:], _pytorch_reference(f2, channel_dim=1), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bad_dim", [0, 2, 4, -2, -4, -5])
    def test_rejects_invalid_channel_dim(self, device, bad_dim):
        """Only ``channel_dim`` resolving to 1 or 3 is accepted for 4D input."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.zeros((2, 4, 4, 3), dtype=torch.uint8, device=device)
        with pytest.raises(ValueError, match="channel_dim must resolve to 1 .BCHW. or 3 .BHWC."):
            normalize_image_uint8(src, channel_dim=bad_dim)
