# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch
import warp as wp

from .proxy_array import ProxyArray

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
    scratch_mask: wp.array | None = None,
    device: str,
) -> wp.array:
    """Resolve ids/mask into a warp boolean mask.

    Callers provide pre-allocated ``all_mask`` (all-True) and ``scratch_mask`` (reusable
    work buffer) so this function never allocates.

    Args:
        ids: Index ids. Accepts ``Sequence[int]``, ``slice``, ``torch.Tensor``,
            ``wp.array(dtype=wp.int32)``, or ``None`` (all elements).
        mask: Direct boolean mask. ``wp.array`` is returned as-is, or copied when on another device;
            ``torch.Tensor`` is converted.
        all_mask: Pre-allocated all-True mask returned when both *ids* and *mask*
            are ``None``.
        scratch_mask: Pre-allocated scratch buffer populated in-place when *ids*
            are provided. Not re-entrant (shared buffer). Defaults to None, for callers
            that never pass *ids*.
        device: Warp device string (e.g. ``"cuda:0"``).

    Returns:
        A ``wp.array(dtype=wp.bool)`` mask.

    Raises:
        ValueError: If *ids* select a subset of elements and no ``scratch_mask`` is given.
    """
    # Normalize slice(None) to None so the capture guard treats it identically to ids=None.
    if isinstance(ids, slice) and ids == slice(None):
        ids = None

    if wp.get_device().is_capturing:
        if ids is not None or (mask is not None and _mask_needs_copy(mask, device)):
            raise RuntimeError(
                "resolve_1d_mask is only capturable when ids are None and mask is None or a boolean mask on device."
            )

    # --- Direct mask input ---
    if mask is not None:
        if isinstance(mask, wp.array):
            if mask.device != wp.get_device(device):
                mask = wp.clone(mask, device=device)
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

    if scratch_mask is None:
        raise ValueError("resolve_1d_mask needs a scratch_mask to write ids into.")

    if isinstance(ids, slice):
        scratch_mask.zero_()
        wp.to_torch(scratch_mask)[ids] = True
        return scratch_mask

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


def _mask_needs_copy(mask: wp.array | torch.Tensor, device: str) -> bool:
    """Whether :func:`resolve_1d_mask` casts or copies ``mask``, which a graph capture cannot record."""
    if isinstance(mask, wp.array):
        return mask.device != wp.get_device(device)
    if isinstance(mask, torch.Tensor):
        return mask.dtype != torch.bool or str(mask.device) != device
    return True


##
# Id resolution - ids/mask to ids.
##


def resolve_ids(
    *,
    ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
    mask: wp.array | torch.Tensor | None = None,
    all_ids: wp.array,
    device: str,
    dtype: type | None = None,
) -> wp.array | torch.Tensor:
    """Resolve ids/mask into ids on ``device``. The mask takes precedence over the ids.

    Args:
        ids: Index ids. Accepts ``Sequence[int]``, ``slice``, ``torch.Tensor``, ``wp.array``,
            or ``None`` (all elements). Tensors and warp arrays keep their integer dtype
            unless ``dtype`` is given. Pass ids from ``find_*(as_proxy=True)`` as ``.warp`` or ``.torch``.
        mask: Boolean mask. Its ids are found with :func:`torch.nonzero`, which waits for
            the device.
        all_ids: Pre-allocated ids of every element, returned as-is when *ids* and *mask*
            select every element.
        device: Warp device string (e.g. ``"cuda:0"``).
        dtype: Warp integer dtype the ids must have, for callers whose kernels take one
            index dtype. The ids are then a contiguous 1-D ``wp.array`` of this dtype.
            Defaults to None: a mask resolves to an int64 ``torch.Tensor``, and ids keep
            their type and dtype.

    Returns:
        ``all_ids``, a ``torch.Tensor`` or a ``wp.array`` of ids on ``device``.

    Raises:
        TypeError: If *ids* is a :class:`ProxyArray`, which is output-only, *mask* is
            neither a ``wp.array`` nor a ``torch.Tensor``, or ``dtype`` is given and the
            ids are not 32-bit or 64-bit integers.
        ValueError: If ``dtype`` is given and the ids are not one-dimensional.
    """
    if mask is not None:
        if isinstance(mask, wp.array):
            mask = wp.to_torch(mask)
        elif not isinstance(mask, torch.Tensor):
            raise TypeError(f"Unsupported mask type: {type(mask)}")
        ids = torch.nonzero(mask.to(wp.device_to_torch(device))).squeeze(-1)
    else:
        ids = _ids_on_device(ids, all_ids, device)
    return ids if dtype is None else _ids_as_dtype(ids, dtype)


def _ids_on_device(
    ids: Sequence[int] | slice | torch.Tensor | wp.array | None, all_ids: wp.array, device: str
) -> wp.array | torch.Tensor:
    """Ids on ``device``, keeping the dtype of tensors and warp arrays; lists and slices become ``wp.int32``."""
    if isinstance(ids, torch.Tensor):
        return ids if str(ids.device) == str(wp.get_device(device)) else ids.to(wp.device_to_torch(device))
    if ids is None:
        return all_ids
    if isinstance(ids, wp.array):
        return ids if ids.device == wp.get_device(device) else ids.to(device)
    if isinstance(ids, ProxyArray):
        raise TypeError("ProxyArray is output-only; pass .warp or .torch explicitly.")
    if isinstance(ids, slice):
        if ids == slice(None):
            return all_ids
        start, stop, step = ids.indices(all_ids.shape[0])
        if step > 0:
            # a strided view of the caller's all-ids, no copy
            return wp.from_torch(wp.to_torch(all_ids)[start:stop:step])
        return wp.array(list(range(start, stop, step)), dtype=wp.int32, device=device)
    return wp.array(list(ids), dtype=wp.int32, device=device)


def _ids_as_dtype(ids: wp.array | torch.Tensor, dtype: type) -> wp.array:
    """``ids`` as a contiguous 1-D ``wp.array`` of ``dtype`` on their device."""
    if dtype not in (wp.int32, wp.int64):
        raise TypeError(f"Unsupported id dtype: {dtype}. Expected wp.int32 or wp.int64.")
    if ids.ndim != 1:
        raise ValueError(f"Ids must be a one-dimensional array, got {ids.ndim} dimensions.")
    if isinstance(ids, wp.array):
        if ids.dtype == dtype and ids.is_contiguous:
            return ids
        if ids.dtype not in (wp.int32, wp.int64):
            raise TypeError(f"Warp ids must be 32-bit or 64-bit signed integers, got {ids.dtype}.")
        ids = wp.to_torch(ids)
    elif ids.dtype.is_floating_point or ids.dtype.is_complex or ids.dtype == torch.bool:
        raise TypeError(f"Ids must be integers, got {ids.dtype}.")
    return wp.from_torch(ids.to(wp.dtype_to_torch(dtype)).contiguous(), dtype=dtype)


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
