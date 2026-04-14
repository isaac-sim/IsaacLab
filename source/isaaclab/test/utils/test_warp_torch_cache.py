# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp

wp.init()

from isaaclab.utils.warp_torch_cache import clear_warp_torch_cache, set_caching_enabled, warp_to_torch


class _Owner:
    """Minimal stub that holds Warp arrays as attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FrozenOwner:
    """Owner that rejects arbitrary attribute writes via __slots__."""

    __slots__ = ("joint_pos",)

    def __init__(self, joint_pos):
        self.joint_pos = joint_pos


class _FakeWarpArray:
    """Lightweight warp-array stand-in for cache-key edge-case tests."""

    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: object, strides: tuple[int, ...], device: object):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.device = device


class _NoWeakrefOwner:
    """Owner that allows cache attr writes but cannot be weak-referenced."""

    __slots__ = ("joint_pos", "_warp_torch_cache")

    def __init__(self, joint_pos):
        self.joint_pos = joint_pos


@pytest.fixture(autouse=True)
def _enable_cache():
    """Ensure caching is enabled during each test and cleared afterwards."""
    set_caching_enabled(True)
    yield
    clear_warp_torch_cache()
    set_caching_enabled(True)


def _make_warp_array(n: int = 4, value: float = 1.0) -> wp.array:
    return wp.from_torch(torch.full((n,), value, dtype=torch.float32, device="cpu"))


# --------------------------------------------------------------------------- #
# Cache hit / identity
# --------------------------------------------------------------------------- #


def test_cache_hit_returns_same_object():
    """Repeated calls with the same Warp buffer return the identical tensor."""
    owner = _Owner(joint_pos=_make_warp_array())
    t1 = warp_to_torch(owner, "joint_pos")
    t2 = warp_to_torch(owner, "joint_pos")
    assert t1 is t2


# --------------------------------------------------------------------------- #
# Zero-copy: in-place mutations visible through cached tensor
# --------------------------------------------------------------------------- #


def test_zero_copy_inplace_update():
    """In-place write to the Warp array is reflected in the cached tensor."""
    arr = _make_warp_array(4, value=0.0)
    owner = _Owner(data=arr)

    cached = warp_to_torch(owner, "data")
    assert cached.sum().item() == 0.0

    arr.numpy()[:] = 7.0
    # Re-fetch — should be same tensor object, now showing updated values.
    cached2 = warp_to_torch(owner, "data")
    assert cached2 is cached
    assert cached.sum().item() == pytest.approx(28.0)


# --------------------------------------------------------------------------- #
# Cache miss on buffer reallocation (pointer change)
# --------------------------------------------------------------------------- #


def test_cache_miss_on_reallocation():
    """Replacing the Warp array with a new buffer triggers a fresh conversion."""
    owner = _Owner(joint_pos=_make_warp_array(4, value=1.0))
    t_old = warp_to_torch(owner, "joint_pos")

    owner.joint_pos = _make_warp_array(4, value=2.0)
    t_new = warp_to_torch(owner, "joint_pos")

    assert t_old is not t_new
    torch.testing.assert_close(t_old, torch.full((4,), 1.0))
    torch.testing.assert_close(t_new, torch.full((4,), 2.0))


# --------------------------------------------------------------------------- #
# Multiple fields on the same owner
# --------------------------------------------------------------------------- #


def test_multiple_fields_independent():
    """Different field names are cached independently on the same owner."""
    owner = _Owner(
        joint_pos=_make_warp_array(3, value=1.0),
        joint_vel=_make_warp_array(3, value=2.0),
    )

    t_pos = warp_to_torch(owner, "joint_pos")
    t_vel = warp_to_torch(owner, "joint_vel")

    assert t_pos is not t_vel
    torch.testing.assert_close(t_pos, torch.full((3,), 1.0))
    torch.testing.assert_close(t_vel, torch.full((3,), 2.0))

    # Each field still cache-hits independently.
    assert warp_to_torch(owner, "joint_pos") is t_pos
    assert warp_to_torch(owner, "joint_vel") is t_vel


# --------------------------------------------------------------------------- #
# Caching disabled
# --------------------------------------------------------------------------- #


def test_disabled_cache_returns_fresh_tensor():
    """With caching off, every call creates a new tensor (still correct data)."""
    set_caching_enabled(False)

    owner = _Owner(joint_pos=_make_warp_array(4, value=3.0))
    t1 = warp_to_torch(owner, "joint_pos")
    t2 = warp_to_torch(owner, "joint_pos")

    assert t1 is not t2
    torch.testing.assert_close(t1, t2)


# --------------------------------------------------------------------------- #
# Frozen / unsettable owner fallback
# --------------------------------------------------------------------------- #


def test_frozen_owner_fallback():
    """Owners that reject __setattr__ still return correct data without error."""
    arr = _make_warp_array(3, value=5.0)
    owner = _FrozenOwner(joint_pos=arr)

    t = warp_to_torch(owner, "joint_pos")
    torch.testing.assert_close(t, torch.full((3,), 5.0))


def test_non_weakref_owner_fallback_uncached():
    """Owners that reject weakref fallback to uncached conversions."""
    owner = _NoWeakrefOwner(joint_pos=_make_warp_array(4, value=5.0))

    t1 = warp_to_torch(owner, "joint_pos")
    t2 = warp_to_torch(owner, "joint_pos")

    assert t1 is not t2
    assert not hasattr(owner, "_warp_torch_cache")
    torch.testing.assert_close(t1, torch.full((4,), 5.0))
    torch.testing.assert_close(t2, torch.full((4,), 5.0))


# --------------------------------------------------------------------------- #
# clear_warp_torch_cache
# --------------------------------------------------------------------------- #


def test_clear_cache_flushes_all_owners():
    """After clear_warp_torch_cache(), subsequent calls return new tensor objects."""
    owner_a = _Owner(joint_pos=_make_warp_array(4, value=1.0))
    owner_b = _Owner(joint_pos=_make_warp_array(4, value=2.0))

    t_a = warp_to_torch(owner_a, "joint_pos")
    t_b = warp_to_torch(owner_b, "joint_pos")

    # Sanity: cache hits before clearing.
    assert warp_to_torch(owner_a, "joint_pos") is t_a
    assert warp_to_torch(owner_b, "joint_pos") is t_b

    clear_warp_torch_cache()

    # After clearing, new tensors are created (same data, different objects).
    t_a2 = warp_to_torch(owner_a, "joint_pos")
    t_b2 = warp_to_torch(owner_b, "joint_pos")
    assert t_a2 is not t_a
    assert t_b2 is not t_b
    torch.testing.assert_close(t_a2, torch.full((4,), 1.0))
    torch.testing.assert_close(t_b2, torch.full((4,), 2.0))


def test_clear_cache_handles_dead_refs():
    """clear_warp_torch_cache() does not crash when owners have been garbage-collected."""
    owner = _Owner(joint_pos=_make_warp_array(4, value=1.0))
    warp_to_torch(owner, "joint_pos")

    del owner
    # Should not raise even though the weak reference is now dead.
    clear_warp_torch_cache()


def test_cache_miss_when_ptr_matches_but_metadata_changes(monkeypatch: pytest.MonkeyPatch):
    """Cache token includes metadata, not only pointer address."""
    owner = _Owner(data=_FakeWarpArray(ptr=0, shape=(0,), dtype="float32", strides=(1,), device="cpu"))

    def _fake_to_torch(arr):
        return torch.empty(arr.shape)

    monkeypatch.setattr(wp, "to_torch", _fake_to_torch)

    t1 = warp_to_torch(owner, "data")

    # Same pointer, different shape/dtype metadata should force a cache miss.
    owner.data = _FakeWarpArray(ptr=0, shape=(0, 3), dtype="int32", strides=(3, 1), device="cpu")
    t2 = warp_to_torch(owner, "data")

    assert t1 is not t2
    assert tuple(t1.shape) == (0,)
    assert tuple(t2.shape) == (0, 3)
