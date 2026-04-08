# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp

wp.init()

from isaaclab.utils.warp_torch_cache import set_caching_enabled, warp_to_torch


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


@pytest.fixture(autouse=True)
def _enable_cache():
    """Ensure caching is enabled before each test and restored afterwards."""
    set_caching_enabled(True)
    yield
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
