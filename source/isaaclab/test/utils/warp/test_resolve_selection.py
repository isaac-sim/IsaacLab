# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolving index selections with :func:`resolve_ids` and :func:`resolve_1d_mask`."""

import pytest
import torch
import warp as wp

from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.utils import resolve_1d_mask, resolve_ids


def _all_ids(n: int, device: str) -> wp.array:
    return wp.array(list(range(n)), dtype=wp.int32, device=device)


@pytest.mark.parametrize("device", test_devices())
def test_resolve_ids(device):
    """Every selection resolves to device ids; a selection of every element is the shared constant itself."""
    all_ids = _all_ids(10, device)
    assert resolve_ids(all_ids=all_ids, device=device) is all_ids
    assert resolve_ids(ids=slice(None), all_ids=all_ids, device=device) is all_ids
    assert resolve_ids(ids=[3, 1], all_ids=all_ids, device=device).numpy().tolist() == [3, 1]
    assert resolve_ids(ids=slice(2, 8, 3), all_ids=all_ids, device=device).numpy().tolist() == [2, 5]
    assert resolve_ids(ids=slice(None, None, -4), all_ids=all_ids, device=device).numpy().tolist() == [9, 5, 1]
    ids_torch = torch.tensor([4, 2], dtype=torch.int64, device=wp.device_to_torch(device))
    assert resolve_ids(ids=ids_torch, all_ids=all_ids, device=device) is ids_torch
    ids_wp = wp.array([6], dtype=wp.int32, device=device)
    assert resolve_ids(ids=ids_wp, all_ids=all_ids, device=device) is ids_wp
    with pytest.raises(TypeError, match="ProxyArray"):
        resolve_ids(ids=ProxyArray(ids_wp), all_ids=all_ids, device=device)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("mask_type", ["warp", "torch_bool", "torch_int"])
def test_resolve_ids_from_a_mask(device, mask_type):
    """A mask resolves to its ascending ids, takes precedence over ids, and honours ``dtype``."""
    n = 4096
    expected = (torch.rand(n, generator=torch.Generator().manual_seed(0)) < 0.5).to(wp.device_to_torch(device))
    mask = {
        "warp": wp.from_torch(expected, dtype=wp.bool),
        "torch_bool": expected,
        "torch_int": expected.to(torch.int64),
    }[mask_type]
    all_ids = _all_ids(n, device)
    ids = resolve_ids(ids=[0, 1], mask=mask, all_ids=all_ids, device=device)
    assert ids.dtype == torch.int64 and torch.equal(ids, torch.nonzero(expected).flatten())
    ids_int32 = resolve_ids(mask=mask, all_ids=all_ids, device=device, dtype=wp.int32)
    assert isinstance(ids_int32, wp.array) and ids_int32.dtype == wp.int32 and ids_int32.is_contiguous
    assert torch.equal(wp.to_torch(ids_int32), torch.nonzero(expected).flatten().to(torch.int32))


@pytest.mark.parametrize("device", test_devices())
def test_resolve_ids_with_dtype(device):
    """``dtype=wp.int32`` gives a contiguous one-dimensional int32 Warp array for every kind of ids."""
    all_ids = _all_ids(10, device)
    torch_device = wp.device_to_torch(device)
    assert resolve_ids(all_ids=all_ids, device=device, dtype=wp.int32) is all_ids
    for ids, expected in (
        (torch.tensor([4, 2], dtype=torch.int64, device=torch_device), [4, 2]),
        (torch.arange(10, device=torch_device)[::3], [0, 3, 6, 9]),
        (wp.array([7, 1], dtype=wp.int64, device=device), [7, 1]),
        (slice(1, 8, 3), [1, 4, 7]),
        (slice(None, None, -4), [9, 5, 1]),
        ([3, 8], [3, 8]),
    ):
        resolved = resolve_ids(ids=ids, all_ids=all_ids, device=device, dtype=wp.int32)
        assert isinstance(resolved, wp.array) and resolved.dtype == wp.int32 and resolved.is_contiguous
        assert resolved.numpy().tolist() == expected
    ids_int32 = wp.array([5], dtype=wp.int32, device=device)
    assert resolve_ids(ids=ids_int32, all_ids=all_ids, device=device, dtype=wp.int32) is ids_int32
    with pytest.raises(TypeError):
        resolve_ids(ids=torch.tensor([1.0], device=torch_device), all_ids=all_ids, device=device, dtype=wp.int32)
    with pytest.raises(TypeError):
        resolve_ids(
            ids=wp.array([1.0], dtype=wp.float32, device=device), all_ids=all_ids, device=device, dtype=wp.int32
        )
    with pytest.raises(ValueError):
        resolve_ids(
            ids=torch.zeros((2, 2), dtype=torch.int64, device=torch_device),
            all_ids=all_ids,
            device=device,
            dtype=wp.int32,
        )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("ids_type", ["list", "slice", "torch_int64", "torch_int32", "warp_int32"])
def test_resolve_1d_mask_from_ids(device, ids_type):
    """Ids of every type are written into the scratch mask, which is cleared first."""
    n = 12
    all_mask = wp.ones(n, dtype=wp.bool, device=device)
    scratch = wp.ones(n, dtype=wp.bool, device=device)
    selected = [1, 4, 5, 11]
    torch_device = wp.device_to_torch(device)
    ids = {
        "list": selected,
        "slice": slice(1, 12, 3),
        "torch_int64": torch.tensor(selected, dtype=torch.int64, device=torch_device),
        "torch_int32": torch.tensor(selected, dtype=torch.int32, device=torch_device),
        "warp_int32": wp.array(selected, dtype=wp.int32, device=device),
    }[ids_type]
    expected = [False] * n
    for i in range(n)[ids] if isinstance(ids, slice) else selected:
        expected[i] = True
    mask = resolve_1d_mask(ids=ids, all_mask=all_mask, scratch_mask=scratch, device=device)
    assert mask is scratch
    assert mask.numpy().tolist() == expected


@pytest.mark.parametrize("device", test_devices())
def test_resolve_1d_mask_needs_a_scratch_mask_only_for_ids(device):
    """A mask or no selection resolves without a scratch mask; ids without one raise."""
    all_mask = wp.ones(5, dtype=wp.bool, device=device)
    mask = wp.array([True, False, True, False, False], dtype=wp.bool, device=device)
    assert resolve_1d_mask(ids=[0], mask=mask, all_mask=all_mask, device=device) is mask
    assert resolve_1d_mask(all_mask=all_mask, device=device) is all_mask
    assert resolve_1d_mask(ids=slice(None), all_mask=all_mask, device=device) is all_mask
    mask_torch = torch.tensor([0, 1, 1, 0, 0], device=wp.device_to_torch(device))
    resolved = resolve_1d_mask(mask=mask_torch, all_mask=all_mask, device=device)
    assert resolved.dtype == wp.bool and resolved.numpy().tolist() == [False, True, True, False, False]
    with pytest.raises(ValueError, match="scratch_mask"):
        resolve_1d_mask(ids=[1], all_mask=all_mask, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_resolve_1d_mask_copies_a_mask_on_another_device():
    """A Warp mask on another device is copied to ``device``."""
    all_mask = wp.ones(3, dtype=wp.bool, device="cuda:0")
    mask = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    resolved = resolve_1d_mask(mask=mask, all_mask=all_mask, device="cuda:0")
    assert str(resolved.device) == "cuda:0" and resolved.numpy().tolist() == [True, False, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_resolve_1d_mask_in_a_graph_capture():
    """Inside a capture, a boolean mask on device resolves; ids or a mask that needs a cast or copy raise."""
    device = "cuda:0"
    all_mask = wp.ones(3, dtype=wp.bool, device=device)
    scratch = wp.zeros(3, dtype=wp.bool, device=device)
    mask_wp = wp.array([True, False, True], dtype=wp.bool, device=device)
    mask_torch = torch.tensor([False, True, True], device=device)
    mask_int = mask_torch.int()
    mask_cpu = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    with wp.ScopedDevice(device), wp.ScopedCapture():
        assert resolve_1d_mask(mask=mask_wp, all_mask=all_mask, device=device) is mask_wp
        resolved = resolve_1d_mask(mask=mask_torch, all_mask=all_mask, device=device)
        for selection in ({"mask": mask_int}, {"mask": mask_cpu}, {"ids": [0]}):
            with pytest.raises(RuntimeError, match="capturable"):
                resolve_1d_mask(**selection, all_mask=all_mask, scratch_mask=scratch, device=device)
    assert resolved.numpy().tolist() == [False, True, True]
