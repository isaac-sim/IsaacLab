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

    def test_multi_tile_h_axis(self, device):
        """H > tile_size must produce NUM_TILES > 1 partial sums and reduce correctly."""
        from isaaclab.utils.warp.ops import _UINT8_SUM_TILE_HW, normalize_image_uint8

        # H = 2 * TILE forces NUM_TILES=2 (evenly split).
        h = 2 * _UINT8_SUM_TILE_HW
        torch.manual_seed(2)
        src = torch.randint(0, 255, (2, h, 8, 6), dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src)
        torch.testing.assert_close(out, _pytorch_reference(src), atol=1e-5, rtol=1e-5)

    def test_non_divisible_h_last_tile_clamps(self, device):
        """H not divisible by tile_size: last tile's row range must clamp via ``wp.min``."""
        from isaaclab.utils.warp.ops import _UINT8_SUM_TILE_HW, normalize_image_uint8

        # H = 2*TILE + 1 forces a 3rd tile with a single row.
        h = 2 * _UINT8_SUM_TILE_HW + 1
        torch.manual_seed(3)
        src = torch.randint(0, 255, (2, h, 8, 6), dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src)
        torch.testing.assert_close(out, _pytorch_reference(src), atol=1e-5, rtol=1e-5)

    def test_bchw_multi_tile_h_axis(self, device):
        """H > tile_size on BCHW (H at axis 2) must reduce correctly across multiple tiles."""
        from isaaclab.utils.warp.ops import _UINT8_SUM_TILE_HW, normalize_image_uint8

        h = 2 * _UINT8_SUM_TILE_HW + 1
        torch.manual_seed(6)
        src = torch.randint(0, 255, (2, 6, h, 8), dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src, channel_dim=1)
        torch.testing.assert_close(out, _pytorch_reference(src, channel_dim=1), atol=1e-5, rtol=1e-5)

    def test_bchw_constant_input(self, device):
        """A constant-valued BCHW uint8 input must normalize to all zeros."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.full((2, 6, 4, 4), 128, dtype=torch.uint8, device=device)
        out = normalize_image_uint8(src, channel_dim=1)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_bchw_preallocated_output_reused(self, device):
        """``out`` kwarg on the BCHW path is written-into and the same object is returned."""
        from isaaclab.utils.warp.ops import normalize_image_uint8

        src = torch.randint(0, 255, (2, 6, 8, 8), dtype=torch.uint8, device=device)
        out = torch.empty(src.shape, dtype=torch.float32, device=device)
        ptr_before = out.data_ptr()
        result = normalize_image_uint8(src, channel_dim=1, out=out)
        assert result is out
        assert result.data_ptr() == ptr_before
        torch.testing.assert_close(result, _pytorch_reference(src, channel_dim=1), atol=1e-5, rtol=1e-5)

        src2 = torch.randint(0, 255, (2, 6, 8, 8), dtype=torch.uint8, device=device)
        result2 = normalize_image_uint8(src2, channel_dim=1, out=out)
        assert result2.data_ptr() == ptr_before
        torch.testing.assert_close(result2, _pytorch_reference(src2, channel_dim=1), atol=1e-5, rtol=1e-5)

    def test_partials_cache_keyed_by_channel_dim(self, device):
        """BCHW and BHWC inputs of identical shape must land in separate cache slots."""
        from isaaclab.utils.warp import ops as warp_ops

        shape = (2, 6, 8, 8)  # ambiguous shape: works as both BHWC (B=2,H=6,W=8,C=8) and BCHW (B=2,C=6,H=8,W=8)
        src_bhwc = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
        src_bchw = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)

        warp_ops._uint8_sum_partials_cache.clear()

        warp_ops.normalize_image_uint8(src_bhwc)  # channel_dim defaults to -1 -> resolves to 3
        warp_ops.normalize_image_uint8(src_bchw, channel_dim=1)

        # Both calls had shape ``shape`` but different channel_dim, so the cache key
        # (shape, device, channel_dim) must distinguish them: two entries, not one.
        assert (shape, device, 3) in warp_ops._uint8_sum_partials_cache
        assert (shape, device, 1) in warp_ops._uint8_sum_partials_cache
        assert len(warp_ops._uint8_sum_partials_cache) == 2

    def test_partials_cache_reuses_scratch_across_calls(self, device):
        """Repeat calls with the same shape must hit the partials cache, not grow it."""
        from isaaclab.utils.warp import ops as warp_ops

        shape = (2, warp_ops._UINT8_SUM_TILE_HW + 4, 8, 6)
        src_a = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
        src_b = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)

        # Clear the cache to isolate this test from prior tests' state.
        warp_ops._uint8_sum_partials_cache.clear()

        warp_ops.normalize_image_uint8(src_a)
        size_after_first = len(warp_ops._uint8_sum_partials_cache)
        first_scratch_ptr = warp_ops._uint8_sum_partials_cache[(shape, device, 3)].data_ptr()

        warp_ops.normalize_image_uint8(src_b)
        size_after_second = len(warp_ops._uint8_sum_partials_cache)
        second_scratch_ptr = warp_ops._uint8_sum_partials_cache[(shape, device, 3)].data_ptr()

        assert size_after_first == 1
        assert size_after_second == size_after_first
        assert first_scratch_ptr == second_scratch_ptr
