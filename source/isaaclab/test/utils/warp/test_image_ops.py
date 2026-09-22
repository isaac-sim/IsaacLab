# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`isaaclab.utils.warp.ops.normalize_image_uint8`."""

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp import ops as warp_ops
from isaaclab.utils.warp.ops import _UINT8_SUM_TILE_HW, normalize_image_uint8

pytestmark = pytest.mark.unit


def _pytorch_reference(src: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    x = src.float() / 255.0
    spatial_dims = tuple(d for d in range(1, x.ndim) if d != channel_dim % x.ndim)
    return x - x.mean(dim=spatial_dims, keepdim=True)


@pytest.fixture(params=test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def device(request):
    return request.param


@pytest.fixture(params=[-1, 1], ids=["bhwc", "bchw"])
def channel_dim(request):
    return request.param


def test_constant_input(device, channel_dim):
    src = torch.full((2, 4, 4, 6), 128, dtype=torch.uint8, device=device)
    out = normalize_image_uint8(src, channel_dim=channel_dim)
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.parametrize("height", [16, 2 * _UINT8_SUM_TILE_HW, 2 * _UINT8_SUM_TILE_HW + 1])
def test_matches_pytorch_reference(device, channel_dim, height):
    shape = (3, 6, height, 8) if channel_dim == 1 else (3, height, 8, 6)
    src = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
    out = normalize_image_uint8(src, channel_dim=channel_dim)
    torch.testing.assert_close(out, _pytorch_reference(src, channel_dim), atol=1e-5, rtol=1e-5)


def test_disjoint_channel_slices(device, channel_dim):
    shape = (2, 3, 8, 8) if channel_dim == 1 else (2, 8, 8, 3)
    frames = [torch.randint(0, 255, shape, dtype=torch.uint8, device=device) for _ in range(2)]
    out = normalize_image_uint8(torch.cat(frames, dim=channel_dim), channel_dim=channel_dim)
    for actual, frame in zip(out.chunk(2, dim=channel_dim), frames):
        torch.testing.assert_close(actual, _pytorch_reference(frame, channel_dim), atol=1e-5, rtol=1e-5)


def test_preallocated_output_reused(device, channel_dim):
    shape = (2, 6, 8, 8) if channel_dim == 1 else (2, 8, 8, 6)
    out = torch.empty(shape, dtype=torch.float32, device=device)
    ptr = out.data_ptr()
    for _ in range(2):
        src = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
        assert normalize_image_uint8(src, channel_dim=channel_dim, out=out) is out
        assert out.data_ptr() == ptr
        torch.testing.assert_close(out, _pytorch_reference(src, channel_dim), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("shape", "dtype"), [((2, 4, 4, 3), torch.float32), ((4, 4, 3), torch.uint8)])
def test_rejects_invalid_input(device, shape, dtype):
    src = torch.zeros(shape, dtype=dtype, device=device)
    with pytest.raises(ValueError, match="4D uint8"):
        normalize_image_uint8(src)


def test_rejects_non_contiguous_input(device):
    src = torch.zeros((2, 8, 8, 12), dtype=torch.uint8, device=device)[..., ::2]
    assert not src.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        normalize_image_uint8(src)


@pytest.mark.parametrize(("shape", "dtype"), [((2, 4, 4, 3), torch.float32), ((2, 4, 4, 6), torch.float16)])
def test_rejects_invalid_output(device, shape, dtype):
    src = torch.zeros((2, 4, 4, 6), dtype=torch.uint8, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    with pytest.raises(ValueError, match="out shape/dtype/device"):
        normalize_image_uint8(src, out=out)


def test_negative_channel_index_matches_positive(device, channel_dim):
    src = torch.randint(0, 255, (2, 4, 8, 8), dtype=torch.uint8, device=device)
    positive = channel_dim % src.ndim
    out_pos = normalize_image_uint8(src, channel_dim=positive)
    out_neg = normalize_image_uint8(src, channel_dim=positive - src.ndim)
    torch.testing.assert_close(out_pos, out_neg)


@pytest.mark.parametrize("bad_dim", [0, 2, 4, -2, -4, -5])
def test_rejects_invalid_channel_dim(device, bad_dim):
    src = torch.zeros((2, 4, 4, 3), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="channel_dim must resolve to 1 .BCHW. or 3 .BHWC."):
        normalize_image_uint8(src, channel_dim=bad_dim)


def test_partials_cache_distinguishes_layout_and_reuses_storage(device, monkeypatch):
    # The same shape is valid in both layouts but needs different scratch storage.
    shape = (2, _UINT8_SUM_TILE_HW + 4, 8, 6)
    monkeypatch.setattr(warp_ops, "_uint8_sum_partials_cache", {})
    scratch = {}
    for _ in range(2):
        for channel_dim in (1, 3):
            src = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
            out = normalize_image_uint8(src, channel_dim=channel_dim)
            torch.testing.assert_close(out, _pytorch_reference(src, channel_dim), atol=1e-5, rtol=1e-5)
            ptr = warp_ops._uint8_sum_partials_cache[(shape, device, channel_dim)].data_ptr()
            assert ptr == scratch.setdefault(channel_dim, ptr)
    assert len(warp_ops._uint8_sum_partials_cache) == 2
