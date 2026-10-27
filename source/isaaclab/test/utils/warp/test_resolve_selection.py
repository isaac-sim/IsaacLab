# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolving index selections with :func:`resolve_ids` and :func:`resolve_1d_mask`."""

import pytest
import torch
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.utils import resolve_1d_mask, resolve_ids


def _all_ids(n: int, device: str) -> wp.array:
    return wp.array(list(range(n)), dtype=wp.int32, device=device)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("dtype", [None, wp.int32])
def test_resolve_ids(device, dtype):
    """Every selection resolves to device ids; a selection of every element is the shared constant itself.

    Without ``dtype``, tensors and Warp arrays pass through and positive-step slices view the shared ids without a
    copy. ``dtype=wp.int32`` gives a contiguous one-dimensional int32 Warp array for every kind of ids.
    """
    all_ids = _all_ids(10, device)
    torch_device = wp.device_to_torch(device)
    ids_int32 = wp.array([5], dtype=wp.int32, device=device)
    for ids in (None, slice(None)):
        assert resolve_ids(ids=ids, all_ids=all_ids, device=device, dtype=dtype) is all_ids
    assert resolve_ids(ids=ids_int32, all_ids=all_ids, device=device, dtype=dtype) is ids_int32
    for ids, expected in (
        ([3, 8], [3, 8]),
        (slice(2, 8, 3), [2, 5]),
        (slice(None, None, -4), [9, 5, 1]),
        (torch.tensor([4, 2], dtype=torch.int64, device=torch_device), [4, 2]),
        (torch.arange(10, device=torch_device)[::3], [0, 3, 6, 9]),
        (wp.array([7, 1], dtype=wp.int64, device=device), [7, 1]),
    ):
        resolved = resolve_ids(ids=ids, all_ids=all_ids, device=device, dtype=dtype)
        if dtype is None:
            assert resolved is ids or isinstance(ids, (list, slice))
        else:
            assert isinstance(resolved, wp.array) and resolved.dtype == dtype and resolved.is_contiguous
        assert (wp.to_torch(resolved) if isinstance(resolved, wp.array) else resolved).tolist() == expected
    with pytest.raises(TypeError, match="ProxyArray"):
        resolve_ids(ids=ProxyArray(ids_int32), all_ids=all_ids, device=device, dtype=dtype)
    if dtype is None:
        cached = wp.to_torch(all_ids)
        for selection in (slice(1, None, 2), slice(0, 0)):
            resolved = wp.to_torch(resolve_ids(ids=selection, all_ids=all_ids, device=device))
            assert resolved.data_ptr() == cached[selection].data_ptr()
            assert resolved.stride() == cached[selection].stride()
    else:
        for ids, error in (
            (torch.tensor([1.0], device=torch_device), TypeError),
            (wp.array([1.0], dtype=wp.float32, device=device), TypeError),
            (torch.zeros((2, 2), dtype=torch.int64, device=torch_device), ValueError),
        ):
            with pytest.raises(error):
                resolve_ids(ids=ids, all_ids=all_ids, device=device, dtype=dtype)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("mask_type", ["warp", "torch"])
def test_resolve_ids_from_a_mask(device, mask_type):
    """A mask resolves to its ascending ids, takes precedence over ids, and honours ``dtype``."""
    mask_bool = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool, device=wp.device_to_torch(device))
    mask = wp.from_torch(mask_bool, dtype=wp.bool) if mask_type == "warp" else mask_bool
    all_ids = _all_ids(mask_bool.shape[0], device)
    ids = resolve_ids(ids=[0, 1], mask=mask, all_ids=all_ids, device=device)
    assert ids.dtype == torch.int64 and ids.tolist() == [1, 2, 4, 7]
    ids_int32 = resolve_ids(mask=mask, all_ids=all_ids, device=device, dtype=wp.int32)
    assert isinstance(ids_int32, wp.array) and ids_int32.dtype == wp.int32 and ids_int32.is_contiguous
    assert ids_int32.numpy().tolist() == [1, 2, 4, 7]


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "ids_type", ["list", "slice", "reverse_slice", "torch_int64", "torch_int32", "warp_int32", "warp_int64"]
)
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
        "reverse_slice": slice(None, None, -2),
        "torch_int64": torch.tensor(selected, dtype=torch.int64, device=torch_device),
        "torch_int32": torch.tensor(selected, dtype=torch.int32, device=torch_device),
        "warp_int32": wp.array(selected, dtype=wp.int32, device=device),
        "warp_int64": wp.array(selected, dtype=wp.int64, device=device),
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
    mask_torch = torch.tensor([False, True, True, False, False], device=wp.device_to_torch(device))
    resolved = resolve_1d_mask(mask=mask_torch, all_mask=all_mask, device=device)
    assert resolved.dtype == wp.bool and resolved.numpy().tolist() == [False, True, True, False, False]
    with pytest.raises(ValueError, match="scratch_mask"):
        resolve_1d_mask(ids=[1], all_mask=all_mask, device=device)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_resolve_1d_mask_copies_a_selection_on_another_device(device, monkeypatch):
    """A Warp mask or Warp ids on another device are copied to ``device``.

    The strict launch mode rejects a kernel argument on another device, which the default mode may read through
    host-memory addressing or fault on.
    """
    monkeypatch.setattr(wp.config, "launch_array_access_mode", wp.config.LaunchArrayAccessMode.STRICT)
    all_mask = wp.ones(3, dtype=wp.bool, device=device)
    scratch = wp.zeros(3, dtype=wp.bool, device=device)
    mask = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    resolved = resolve_1d_mask(mask=mask, all_mask=all_mask, device=device)
    assert str(resolved.device) == device and resolved.numpy().tolist() == [True, False, True]
    ids = wp.array([0, 2], dtype=wp.int32, device="cpu")
    assert resolve_1d_mask(ids=ids, all_mask=all_mask, scratch_mask=scratch, device=device) is scratch
    assert scratch.numpy().tolist() == [True, False, True]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_resolve_1d_mask_in_a_graph_capture(device):
    """Inside a capture, a boolean mask on device resolves; ids or a mask that needs a cast or copy raise.

    ``"cuda"`` names the current CUDA device, so a mask on it needs no copy.
    """
    all_mask = wp.ones(3, dtype=wp.bool, device=device)
    scratch = wp.zeros(3, dtype=wp.bool, device=device)
    mask_wp = wp.array([True, False, True], dtype=wp.bool, device=device)
    mask_torch = torch.tensor([False, True, True], device=device)
    mask_int = mask_torch.int()
    mask_cpu = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    with wp.ScopedDevice(device), wp.ScopedCapture():
        assert resolve_1d_mask(mask=mask_wp, all_mask=all_mask, device=device) is mask_wp
        resolved = resolve_1d_mask(mask=mask_torch, all_mask=all_mask, device=device)
        assert resolve_1d_mask(mask=mask_torch, all_mask=all_mask, device="cuda").ptr == resolved.ptr
        for selection in ({"mask": mask_int}, {"mask": mask_cpu}, {"ids": [0]}):
            with pytest.raises(RuntimeError, match="capturable"):
                resolve_1d_mask(**selection, all_mask=all_mask, scratch_mask=scratch, device=device)
    assert resolved.numpy().tolist() == [False, True, True]
