# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers to reuse :func:`warp.to_torch` views across hot paths (e.g. observation terms).

Because :func:`warp.to_torch` returns a zero-copy tensor that shares the
underlying GPU buffer, in-place modifications to the Warp array are
automatically visible through the cached tensor.  The cache only needs to
be invalidated when the Warp array is **reassigned** to a different buffer
(i.e. its memory pointer changes).

All caches are automatically flushed on :attr:`~isaaclab.physics.PhysicsEvent.STOP`
so that no stale tensor views survive a simulation teardown or scene change.
"""

from __future__ import annotations

import weakref

import torch
import warp as wp

_CACHING_ENABLED: bool = True
_CACHED_OWNERS: list[weakref.ref] = []


def clear_warp_torch_cache() -> None:
    """Clear all cached :func:`warp.to_torch` tensors across every tracked owner.

    This is registered as a :attr:`~isaaclab.physics.PhysicsEvent.STOP` callback
    by :class:`~isaaclab.sim.SimulationContext` so that stale tensor views do not
    survive a simulation teardown or scene change.  It can also be called manually.
    """
    for ref in _CACHED_OWNERS:
        obj = ref()
        if obj is not None:
            try:
                delattr(obj, "_warp_torch_cache")
            except (AttributeError, TypeError):
                pass
    _CACHED_OWNERS.clear()


def set_caching_enabled(enabled: bool) -> None:
    """Enable or disable the :func:`warp.to_torch` cache.

    Called by :class:`~isaaclab.sim.SimulationContext` at init based on
    :attr:`~isaaclab.sim.SimulationCfg.enable_warp_torch_cache`.  Flushes
    all existing cached entries when the setting changes.

    Args:
        enabled: Whether to cache ``wp.to_torch()`` views.
    """
    global _CACHING_ENABLED
    if enabled != _CACHING_ENABLED:
        clear_warp_torch_cache()
    _CACHING_ENABLED = enabled


def warp_to_torch(owner: object, field_name: str) -> torch.Tensor:
    """Return a cached PyTorch tensor view of a Warp-backed attribute on *owner*.

    Resolves ``getattr(owner, field_name)`` and caches the resulting
    :func:`warp.to_torch` tensor on *owner*.  The cached tensor is reused as
    long as the Warp array's identity token (pointer, shape, dtype, strides,
    and device) is unchanged.
    Since ``wp.to_torch`` produces a zero-copy view, in-place updates to the
    Warp buffer are automatically reflected in the cached tensor without
    invalidation.

    Caching can be disabled globally via
    :attr:`~isaaclab.sim.SimulationCfg.enable_warp_torch_cache`.

    Args:
        owner: The object that owns the Warp-backed property.
        field_name: Attribute name on *owner* that returns a
            :class:`wp.array` (e.g. ``"joint_pos"``, ``"root_pos_w"``).

    Returns:
        PyTorch tensor sharing memory with the underlying Warp buffer.
    """
    if not _CACHING_ENABLED:
        return wp.to_torch(getattr(owner, field_name))

    try:
        cache = getattr(owner, "_warp_torch_cache", None)
        if cache is None:
            try:
                owner_ref = weakref.ref(owner)
            except TypeError:
                # Owner cannot be weak-referenced, so we cannot guarantee global
                # cache clearing on STOP. Fall back to uncached conversion.
                return wp.to_torch(getattr(owner, field_name))
            cache = {}
            object.__setattr__(owner, "_warp_torch_cache", cache)
            _CACHED_OWNERS.append(owner_ref)
    except (AttributeError, TypeError):
        return wp.to_torch(getattr(owner, field_name))

    warp_array = getattr(owner, field_name)
    current_token = (
        warp_array.ptr,
        warp_array.shape,
        warp_array.dtype,
        warp_array.strides,
        warp_array.device,
    )

    entry = cache.get(field_name)
    if entry is not None and entry[0] == current_token:
        return entry[1]

    new_tensor = wp.to_torch(warp_array)
    cache[field_name] = (current_token, new_tensor)
    return new_tensor
