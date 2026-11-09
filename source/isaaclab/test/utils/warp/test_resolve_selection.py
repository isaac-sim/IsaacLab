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
from isaaclab.utils.warp.memory import resolve_1d_indices, resolve_1d_mask, resolve_ids


def _all_ids(n: int, device: str) -> wp.array:
    return wp.array(list(range(n)), dtype=wp.int32, device=device)


@pytest.mark.parametrize("device", test_devices())
def test_resolve_ids(device):
    """Every selection resolves to device ids; a selection of every element is the shared constant itself.

    Tensors and Warp arrays pass through, and positive-step slices view the shared ids without a copy.
    """
    all_ids = _all_ids(10, device)
    torch_device = wp.device_to_torch(device)
    ids_int32 = wp.array([5], dtype=wp.int32, device=device)
    for ids in (None, slice(None)):
        assert resolve_ids(ids=ids, all_ids=all_ids) is all_ids
    assert resolve_ids(ids=ids_int32, all_ids=all_ids) is ids_int32
    for ids, expected in (
        ([3, 8], [3, 8]),
        (slice(2, 8, 3), [2, 5]),
        (slice(None, None, -4), [9, 5, 1]),
        (torch.tensor([4, 2], dtype=torch.int64, device=torch_device), [4, 2]),
        (torch.arange(10, device=torch_device)[::3], [0, 3, 6, 9]),
        (wp.array([7, 1], dtype=wp.int64, device=device), [7, 1]),
    ):
        resolved = resolve_ids(ids=ids, all_ids=all_ids)
        assert resolved is ids or isinstance(ids, (list, slice))
        assert (wp.to_torch(resolved) if isinstance(resolved, wp.array) else resolved).tolist() == expected
    cached = wp.to_torch(all_ids)
    for selection in (slice(1, None, 2), slice(0, 0)):
        resolved = wp.to_torch(resolve_ids(ids=selection, all_ids=all_ids))
        assert resolved.data_ptr() == cached[selection].data_ptr()
        assert resolved.stride() == cached[selection].stride()


@pytest.mark.parametrize("device", test_devices())
def test_proxy_array_selects_through_its_warp_array(device):
    """Every resolver takes a ProxyArray selector as its Warp array, without a copy."""
    ids = wp.array([3, 1], dtype=wp.int32, device=device)
    mask = wp.array([0, 1, 0, 1, 0], dtype=wp.bool, device=device)
    all_ids = _all_ids(5, device)
    assert resolve_ids(ids=ProxyArray(ids), all_ids=all_ids) is ids
    assert resolve_ids(mask=ProxyArray(mask), all_ids=all_ids).tolist() == [1, 3]
    assert resolve_1d_indices(ProxyArray(ids)).ptr == ids.ptr
    all_mask = wp.ones(5, dtype=wp.bool, device=device)
    assert resolve_1d_mask(mask=ProxyArray(mask), all_mask=all_mask) is mask
    scratch = wp.zeros(5, dtype=wp.bool, device=device)
    assert resolve_1d_mask(ids=ProxyArray(ids), all_mask=all_mask, scratch_mask=scratch).numpy().tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("mask_type", ["warp", "torch"])
def test_resolve_ids_from_a_mask(device, mask_type):
    """A mask resolves to its ascending ids when no ids are given, and given ids are used as given;
    :func:`resolve_1d_indices` casts to int32."""
    mask_bool = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool, device=wp.device_to_torch(device))
    mask = wp.from_torch(mask_bool, dtype=wp.bool) if mask_type == "warp" else mask_bool
    all_ids = _all_ids(mask_bool.shape[0], device)
    assert resolve_ids(ids=[0, 1], mask=mask, all_ids=all_ids).numpy().tolist() == [0, 1]
    for given in (None, slice(None)):
        ids = resolve_ids(ids=given, mask=mask, all_ids=all_ids)
        assert ids.dtype == torch.int64 and ids.tolist() == [1, 2, 4, 7]
    ids_int32 = resolve_1d_indices(resolve_ids(mask=mask, all_ids=all_ids))
    assert isinstance(ids_int32, wp.array) and ids_int32.dtype == wp.int32 and ids_int32.is_contiguous
    assert ids_int32.numpy().tolist() == [1, 2, 4, 7]


@pytest.mark.parametrize("device", test_devices())
def test_resolve_1d_mask_needs_a_scratch_mask_only_for_ids(device):
    """A mask or no selection resolves without a scratch mask; ids without one raise."""
    all_mask = wp.ones(5, dtype=wp.bool, device=device)
    mask = wp.array([True, False, True, False, False], dtype=wp.bool, device=device)
    assert resolve_1d_mask(ids=[0], mask=mask, all_mask=all_mask) is mask
    assert resolve_1d_mask(all_mask=all_mask) is all_mask
    assert resolve_1d_mask(ids=slice(None), all_mask=all_mask) is all_mask
    mask_torch = torch.tensor([False, True, True, False, False], device=wp.device_to_torch(device))
    resolved = resolve_1d_mask(mask=mask_torch, all_mask=all_mask)
    assert resolved.dtype == wp.bool and resolved.numpy().tolist() == [False, True, True, False, False]
    with pytest.raises(ValueError, match="scratch_mask"):
        resolve_1d_mask(ids=[1], all_mask=all_mask)


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
    resolved = resolve_1d_mask(mask=mask, all_mask=all_mask)
    assert str(resolved.device) == device and resolved.numpy().tolist() == [True, False, True]
    ids = wp.array([0, 2], dtype=wp.int32, device="cpu")
    assert resolve_1d_mask(ids=ids, all_mask=all_mask, scratch_mask=scratch) is scratch
    assert scratch.numpy().tolist() == [True, False, True]


@pytest.mark.parametrize("device", test_devices())
def test_resolve_ids_for_torch_indexing(device):
    """With ``all_ids=slice(None)``, every element stays a slice, given ids are returned as given, and a mask
    resolves to its ids."""
    mask = wp.array([False, True, True, False], dtype=wp.bool, device=device)
    ids = torch.tensor([3, 0], device=wp.device_to_torch(device))
    assert resolve_ids(all_ids=slice(None)) == slice(None)
    assert resolve_ids(ids=slice(1, 3), all_ids=slice(None)) == slice(1, 3)
    assert resolve_ids(ids=ids, mask=mask, all_ids=slice(None)) is ids
    assert resolve_ids(ids=slice(None), mask=mask, all_ids=slice(None)).tolist() == [1, 2]
