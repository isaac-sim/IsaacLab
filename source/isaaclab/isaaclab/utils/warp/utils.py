# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch
import warp as wp

##
# Mask resolution - ids/mask to warp boolean mask.
##


@wp.kernel
def _populate_mask_from_ids(
    mask: wp.array(dtype=wp.bool),
    ids: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    mask[ids[i]] = True


def resolve_1d_mask(
    *,
    ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
    mask: wp.array | torch.Tensor | None = None,
    all_mask: wp.array,
    scratch_mask: wp.array,
    device: str,
) -> wp.array:
    """Resolve ids/mask into a warp boolean mask.

    Callers provide pre-allocated ``all_mask`` (all-True) and ``scratch_mask`` (reusable
    work buffer) so this function never allocates.

    Args:
        ids: Index ids. Accepts ``Sequence[int]``, ``slice``, ``torch.Tensor``,
            ``wp.array(dtype=wp.int32)``, or ``None`` (all elements).
        mask: Direct boolean mask. ``wp.array`` is returned as-is;
            ``torch.Tensor`` is converted.
        all_mask: Pre-allocated all-True mask returned when both *ids* and *mask*
            are ``None``.
        scratch_mask: Pre-allocated scratch buffer populated in-place when *ids*
            are provided. Not re-entrant (shared buffer).
        device: Warp device string (e.g. ``"cuda:0"``).

    Returns:
        A ``wp.array(dtype=wp.bool)`` mask.
    """
    # Normalize slice(None) to None so the capture guard treats it identically to ids=None.
    if isinstance(ids, slice) and ids == slice(None):
        ids = None

    if wp.get_device().is_capturing:
        if ids is not None or (mask is not None and not isinstance(mask, wp.array)):
            raise RuntimeError(
                "resolve_1d_mask is only capturable when mask is a wp.array or both ids and mask are None."
            )

    # --- Direct mask input ---
    if mask is not None:
        if isinstance(mask, wp.array):
            return mask
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if str(mask.device) != device:
                mask = mask.to(device)
            return wp.from_torch(mask, dtype=wp.bool)
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    # --- Fast path: all elements ---
    if ids is None:
        return all_mask

    # --- Normalize slice to list ---
    if isinstance(ids, slice):
        start, stop, step = ids.indices(scratch_mask.shape[0])
        ids = list(range(start, stop, step))

    # --- Normalize to concrete type ---
    if not isinstance(ids, (torch.Tensor, wp.array)):
        ids = list(ids)

    # --- Populate scratch mask ---
    scratch_mask.fill_(False)

    if isinstance(ids, torch.Tensor):
        if ids.numel() == 0:
            return scratch_mask
        if str(ids.device) != device:
            ids = ids.to(device)
        if ids.dtype != torch.int32:
            ids = ids.to(dtype=torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        ids_wp = wp.from_torch(ids, dtype=wp.int32)
    elif isinstance(ids, wp.array):
        if ids.shape[0] == 0:
            return scratch_mask
        if ids.dtype != wp.int32:
            raise TypeError(f"Unsupported wp.array dtype for ids: {ids.dtype}. Expected wp.int32 index array.")
        ids_wp = ids
    else:
        if len(ids) == 0:
            return scratch_mask
        ids_wp = wp.array(ids, dtype=wp.int32, device=device)

    wp.launch(_populate_mask_from_ids, dim=ids_wp.shape[0], inputs=[scratch_mask, ids_wp], device=device)
    return scratch_mask


##
# Capture safety — property guard.
##


def capture_unsafe(reason: str | None = None):
    """Mark a callable as not CUDA-graph-capture-safe.

    Raises ``RuntimeError`` if the decorated callable is invoked while
    ``wp.get_device().is_capturing`` is ``True``.

    Args:
        reason: Optional explanation appended to the error message.

    Usage::

        @property
        @capture_unsafe("Relies on a Python timestamp guard.")
        def projected_gravity_b(self) -> wp.array: ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if wp.get_device().is_capturing:
                msg = f"'{func.__qualname__}' cannot be called during CUDA graph capture."
                if reason:
                    msg = f"{msg} {reason}"
                raise RuntimeError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator
