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
"""

from __future__ import annotations

import torch
import warp as wp

_CACHING_ENABLED: bool = True


def set_caching_enabled(enabled: bool) -> None:
    """Enable or disable the :func:`warp.to_torch` pointer-based cache.

    Called by :class:`~isaaclab.sim.SimulationContext` at init based on
    :attr:`~isaaclab.sim.SimulationCfg.enable_warp_torch_cache`.

    Args:
        enabled: Whether to cache ``wp.to_torch()`` views.
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = enabled


def warp_to_torch(owner: object, field_name: str) -> torch.Tensor:
    """Return a cached PyTorch tensor view of a Warp-backed attribute on *owner*.

    Resolves ``getattr(owner, field_name)`` and caches the resulting
    :func:`warp.to_torch` tensor on *owner*.  The cached tensor is reused as
    long as the Warp array's memory pointer (:attr:`wp.array.ptr`) is unchanged.
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
            cache = {}
            object.__setattr__(owner, "_warp_torch_cache", cache)
    except (AttributeError, TypeError):
        return wp.to_torch(getattr(owner, field_name))

    warp_array = getattr(owner, field_name)
    current_ptr = int(warp_array.ptr)

    entry = cache.get(field_name)
    if entry is not None and entry[0] == current_ptr:
        return entry[1]

    new_tensor = wp.to_torch(warp_array)
    cache[field_name] = (current_ptr, new_tensor)
    return new_tensor
