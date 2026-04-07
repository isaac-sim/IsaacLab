# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers to reuse :func:`warp.to_torch` views across hot paths (e.g. observation terms)."""

from __future__ import annotations

import torch
import warp as wp

_CACHING_ENABLED: bool = True


def set_caching_enabled(enabled: bool) -> None:
    """Enable or disable the per-step :func:`warp.to_torch` cache.

    Called by :class:`~isaaclab.sim.SimulationContext` at init based on
    :attr:`~isaaclab.sim.SimulationCfg.enable_warp_torch_cache`.

    Args:
        enabled: Whether to cache ``wp.to_torch()`` views.
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = enabled


def warp_to_torch(owner_or_array: object, field_name: str | None = None) -> torch.Tensor:
    """Return a PyTorch tensor view, optionally caching per simulation step.

    Two calling conventions are supported:

    - ``warp_to_torch(owner, "field_name")`` — resolves
      ``getattr(owner, field_name)`` and caches the result on *owner* so at most
      one :func:`warp.to_torch` call is performed per field per step.
    - ``warp_to_torch(warp_array)`` — plain :func:`warp.to_torch` with no caching.

    The cache is invalidated each physics substep via the owner's
    ``_sim_timestamp``, or by a change in the Warp buffer pointer for owners
    that lack a timestamp (e.g. sensor data with fixed Warp storage).

    Caching can be disabled globally via
    :attr:`~isaaclab.sim.SimulationCfg.enable_warp_torch_cache`.

    Args:
        owner_or_array: Either the object that owns the Warp-backed property
            (when *field_name* is given) or a raw :class:`wp.array` to convert
            directly.
        field_name: Attribute name on *owner_or_array* that returns a
            :class:`wp.array` (e.g. ``"joint_pos"``, ``"root_pos_w"``).

    Returns:
        PyTorch tensor sharing memory with the underlying Warp buffer.
    """
    if field_name is None:
        return wp.to_torch(owner_or_array)  # type: ignore[arg-type]

    owner = owner_or_array

    if not _CACHING_ENABLED:
        return wp.to_torch(getattr(owner, field_name))

    try:
        cache = getattr(owner, "_warp_torch_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(owner, "_warp_torch_cache", cache)
    except (AttributeError, TypeError):
        return wp.to_torch(getattr(owner, field_name))

    sim_ts = getattr(owner, "_sim_timestamp", None)

    if sim_ts is not None:
        entry = cache.get(field_name)
        if entry is not None and entry[0] == sim_ts:
            return entry[1]

    warp_array = getattr(owner, field_name)

    if sim_ts is not None:
        cache[field_name] = (sim_ts, wp.to_torch(warp_array))
        return cache[field_name][1]

    ptr = int(warp_array.ptr)
    entry = cache.get(field_name)
    if entry is None or entry[0] != ptr:
        cache[field_name] = (ptr, wp.to_torch(warp_array))
    return cache[field_name][1]
