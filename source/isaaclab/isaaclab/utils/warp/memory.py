# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Value-preserving operations on caller-supplied array layouts and selections.

Callers resolve their attribute frequency into array shapes, strides and selectors.
This module neither names frequencies nor interprets element components. Use native Warp
views, copy and fill for ordinary movement; cached kernels handle masks and fused operations.
Warm up the execution path before graph capture.
"""

from collections.abc import Sequence
from functools import cache
from numbers import Integral

import numpy as np
import torch
import warp as wp

from .index_kernel import selector_dtype

_CONCAT_TILE_SIZE = 32


def _component_layout(dtypes: tuple) -> tuple[type, tuple[int, ...]]:
    """Validate one-dimensional composite elements and return their scalar type and sizes."""
    if not dtypes or any(len(getattr(dtype, "_shape_", ())) != 1 for dtype in dtypes):
        raise TypeError("Expected one-dimensional composite element types")
    scalar_dtype = dtypes[0]._wp_scalar_type_
    if any(dtype._wp_scalar_type_ != scalar_dtype for dtype in dtypes):
        raise TypeError("Component scalar types must match")
    return scalar_dtype, tuple(wp.types.type_size(dtype) for dtype in dtypes)


@cache
def _make_concat(dtype, first_size: int, second_size: int, ndim: int) -> wp.Kernel:
    """Generate a tiled concatenation over scalar views, specializing only the layout."""
    result_size = first_size + second_size

    @wp.kernel(module="unique")
    def concatenate(
        first: wp.array(dtype=dtype, ndim=ndim + 1),
        second: wp.array(dtype=dtype, ndim=ndim + 1),
        result: wp.array(dtype=dtype, ndim=ndim + 1),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            a, b, out = first, second, result
        else:
            row, i = wp.tid()
            a, b, out = first[row], second[row], result[row]
        offset = (i * _CONCAT_TILE_SIZE, 0)
        x = wp.tile_load(a, shape=(_CONCAT_TILE_SIZE, first_size), offset=offset)
        y = wp.tile_load(b, shape=(_CONCAT_TILE_SIZE, second_size), offset=offset)
        value = wp.tile_empty(shape=(_CONCAT_TILE_SIZE, result_size), dtype=dtype, storage="shared")
        wp.tile_assign(value, x, offset=(0, 0))
        wp.tile_assign(value, y, offset=(0, first_size))
        wp.tile_store(out, value, offset=offset)

    return concatenate


def concat(
    first: wp.array, second: wp.array, result: wp.array, *, device=None, record_cmd: bool = False
) -> wp.Launch | None:
    """Concatenate composite elements in one tiled launch without allocating array storage.

    Arrays must have the same shape (rank 1 or 2) and device. Elements must have a
    one-dimensional component layout with matching scalar types. The output width
    must equal the sum of the input widths. Strides are preserved; contiguous rank-2
    arrays share the flattened path. Input and output storage must not overlap.

    Like :func:`warp.launch_tiled`, return a recorded command when ``record_cmd`` is
    true, otherwise launch immediately. Empty arrays launch no work and return None.
    Recorded commands bind stable array storage.
    """
    if first.ndim not in (1, 2):
        raise ValueError(f"Expected rank 1 or 2, got {first.ndim}")
    if first.shape != second.shape or first.shape != result.shape:
        raise ValueError("Concatenation requires matching array shapes")
    if first.device != second.device or first.device != result.device:
        raise ValueError("Concatenation requires matching array devices")
    if device is not None and wp.get_device(device) != result.device:
        raise ValueError("Launch device must match the array device")
    dtype, (first_size, second_size, result_size) = _component_layout((first.dtype, second.dtype, result.dtype))
    if first_size + second_size != result_size:
        raise ValueError("Output component count must equal the sum of the input counts")
    arrays = (first, second, result)
    if first.ndim == 2 and all(array.is_contiguous for array in arrays):
        arrays = tuple(array.reshape(array.size) for array in arrays)
    first, second, result = (array.view(dtype) for array in arrays)
    shape = first.shape[:-1]
    dim = (*shape[:-1], (shape[-1] + _CONCAT_TILE_SIZE - 1) // _CONCAT_TILE_SIZE)
    return wp.launch_tiled(
        _make_concat(dtype, first_size, second_size, len(shape)),
        dim=dim,
        inputs=[first, second],
        outputs=[result],
        device=result.device,
        block_dim=128,
        record_cmd=record_cmd,
    )


@cache
def make_split(first_dtype, second_dtype, ndim: int) -> wp.Kernel:
    """Copy scalar components into two typed outputs, optionally selecting rows.

    Output elements must have a one-dimensional component layout (no scalars or matrices).
    The source has rank ``ndim + 1`` with concatenated components on its last
    axis. Pass ``None`` for the row mask to copy all rows. Unselected rows remain
    untouched. Typed input arrays can supply a scalar ``view`` without a copy.
    """
    if ndim not in (1, 2):
        raise ValueError(f"Expected rank 1 or 2, got {ndim}")
    scalar_dtype, (first_size, second_size) = _component_layout((first_dtype, second_dtype))

    @wp.kernel(module="unique")
    def split(
        source: wp.array(dtype=scalar_dtype, ndim=ndim + 1),
        rows: wp.array(dtype=wp.bool),
        first: wp.array(dtype=first_dtype, ndim=ndim),
        second: wp.array(dtype=second_dtype, ndim=ndim),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
        else:
            i, j = wp.tid()
        if rows and not rows[i]:
            return
        a, b = first_dtype(), second_dtype()
        for k in range(wp.static(first_size)):
            if wp.static(ndim == 1):
                a[k] = source[i, k]
            else:
                a[k] = source[i, j, k]
        for k in range(wp.static(second_size)):
            if wp.static(ndim == 1):
                b[k] = source[i, first_size + k]
            else:
                b[k] = source[i, j, first_size + k]
        if wp.static(ndim == 1):
            first[i] = a
            second[i] = b
        else:
            first[i, j] = a
            second[i, j] = b

    return split


def split_views(source: wp.array, component_dtypes: tuple) -> tuple[wp.array, ...]:
    """Return zero-copy component views with the source's rank and outer strides.

    Components must exactly partition the source element and share its scalar type.
    Rank is limited to 3 because the intermediate scalar view adds one dimension.
    """
    if source.ndim > 3:
        raise ValueError(f"Expected rank at most 3, got ndim={source.ndim}")
    scalar_dtype, (source_size, *sizes) = _component_layout((source.dtype, *component_dtypes))
    if not component_dtypes or sum(sizes) != source_size:
        raise ValueError("Components must exactly partition the source element")
    scalars = source.view(scalar_dtype)
    views, offset = [], 0
    for dtype, size in zip(component_dtypes, sizes, strict=True):
        selection = (slice(None),) * source.ndim + (slice(offset, offset + size),)
        views.append(scalars[selection].view(dtype))
        offset += size
    return tuple(views)


@cache
def make_clear_pair(dtype, ndim: int) -> wp.Kernel:
    """Clear two equally shaped arrays in one launch, selecting their first axis."""
    if ndim not in (1, 2):
        raise ValueError(f"Expected rank 1 or 2, got {ndim}")

    @wp.kernel(module="unique")
    def clear(
        rows: wp.array(dtype=wp.bool), first: wp.array(dtype=dtype, ndim=ndim), second: wp.array(dtype=dtype, ndim=ndim)
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            if rows[i]:
                first[i] = dtype(0.0)
                second[i] = dtype(0.0)
        else:
            i, j = wp.tid()
            if rows[i]:
                first[i, j] = dtype(0.0)
                second[i, j] = dtype(0.0)

    return clear


def resolve_1d_indices(indices: "torch.Tensor | wp.array") -> wp.array:
    """Return contiguous int32 device indices for native Warp indexed views.

    Callers supply valid nonnegative indices into their destination. Warp array
    dimensions are int32, so valid indices fit without narrowing their meaning.
    Contiguous int32 storage is reused; strided or int64 input needs one copy or
    cast. Keep the returned array alive with its indexed view. Capture involving
    a conversion requires Warp's CUDA memory pool and warming the cast beforehand.
    """
    selector_dtype(indices)
    if isinstance(indices, torch.Tensor):
        indices = wp.from_torch(indices)
    if indices.ndim != 1:
        raise ValueError("Indices must be one-dimensional")
    if indices.dtype == wp.int32:
        return indices.contiguous()
    result = wp.empty(indices.shape, dtype=wp.int32, device=indices.device)
    wp.utils.array_cast(indices, result)
    return result


@cache
def make_masked_copy(dtype, ndim: int) -> wp.Kernel:
    """Copy matching elements selected by optional masks on the first two axes.

    Arrays share a shape and may have different strides. A None mask selects the
    whole axis. Use native wp.copy and wp.indexedarray for unmasked/indexed writes.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"Expected rank 1, 2 or 3, got {ndim}")

    @wp.kernel(module="unique")
    def copy(
        source: wp.array(dtype=dtype, ndim=ndim),
        rows: wp.array(dtype=wp.bool),
        columns: wp.array(dtype=wp.bool),
        target: wp.array(dtype=dtype, ndim=ndim),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            j = 0
        elif wp.static(ndim == 2):
            i, j = wp.tid()
        else:
            i, j, k = wp.tid()
        if rows and not rows[i]:
            return
        if wp.static(ndim > 1):
            if columns and not columns[j]:
                return
        if wp.static(ndim == 1):
            target[i] = source[i]
        elif wp.static(ndim == 2):
            target[i, j] = source[i, j]
        else:
            target[i, j, k] = source[i, j, k]

    return copy


@cache
def make_masked_fill(dtype, ndim: int) -> wp.Kernel:
    """Fill elements selected by optional masks on the first two axes."""
    if ndim not in (1, 2, 3):
        raise ValueError(f"Expected rank 1, 2 or 3, got {ndim}")

    @wp.kernel(module="unique")
    def fill(
        value: dtype,
        rows: wp.array(dtype=wp.bool),
        columns: wp.array(dtype=wp.bool),
        target: wp.array(dtype=dtype, ndim=ndim),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            j = 0
        elif wp.static(ndim == 2):
            i, j = wp.tid()
        else:
            i, j, k = wp.tid()
        if rows and not rows[i]:
            return
        if wp.static(ndim > 1):
            if columns and not columns[j]:
                return
        if wp.static(ndim == 1):
            target[i] = value
        elif wp.static(ndim == 2):
            target[i, j] = value
        else:
            target[i, j, k] = value

    return fill


@cache
def _make_populate_mask(dtype) -> wp.Kernel:
    @wp.kernel(module="unique")
    def populate_mask(mask: wp.array(dtype=wp.bool), ids: wp.array(dtype=dtype)):
        i = wp.tid()
        size = wp.int64(mask.shape[0])
        index = wp.int64(ids[i])
        if index < 0:
            index += size
        if index >= 0 and index < size:
            mask[index] = True

    return populate_mask


@wp.kernel
def _slice_mask(mask: wp.array(dtype=wp.bool), start: int, step: int):
    i = wp.tid()
    mask[start + i * step] = True


def resolve_1d_mask(
    *,
    ids: "Sequence[int] | slice | torch.Tensor | wp.array | None" = None,
    mask: "wp.array | torch.Tensor | None" = None,
    all_mask: wp.array,
    scratch_mask: wp.array,
) -> wp.array:
    """Resolve a selection using caller-owned boolean storage on its device.

    An explicit boolean mask takes precedence. ``None`` and ``slice(None)`` return
    the identical ``all_mask`` object. Boolean selectors are masks; signed int32/int64
    selectors are indices, supporting negatives and duplicates. Index values must be
    in ``[-size, size)``; invalid device indices are ignored without a host sync.

    Same-device Warp/Torch arrays and slices are capturable without allocating device
    storage. Host sequences and device transfers are eager-only and may allocate.
    Temporary CUDA index uploads synchronize their stream before releasing storage;
    preallocated device selectors avoid this synchronization.
    Integer widths and strides are preserved. Masks must have the destination shape
    and boolean dtype. The returned scratch storage is reused, so this is not reentrant.
    """
    device = scratch_mask.device
    if scratch_mask.ndim != 1 or scratch_mask.dtype != wp.bool:
        raise ValueError("scratch_mask must be a rank-1 boolean Warp array")
    if all_mask.shape != scratch_mask.shape or all_mask.dtype != wp.bool or all_mask.device != device:
        raise ValueError("all_mask must match scratch_mask shape, dtype and device")

    temporary_input = False
    selection = mask if mask is not None else ids
    if selection is None or (isinstance(selection, slice) and selection == slice(None)):
        return all_mask
    if isinstance(selection, slice):
        start, stop, step = selection.indices(scratch_mask.shape[0])
        scratch_mask.zero_()
        count = len(range(start, stop, step))
        if count:
            wp.launch(_slice_mask, count, [scratch_mask, start, step], device=device)
        return scratch_mask

    if isinstance(selection, torch.Tensor):
        if selection.dtype not in (torch.bool, torch.int32, torch.int64):
            raise TypeError("Selectors must use boolean, int32 or int64 values")
        if wp.device_from_torch(selection.device) != device:
            if device.is_capturing:
                raise RuntimeError("Mask resolution cannot transfer selectors during graph capture")
            selection = selection.to(str(device))
            temporary_input = True
        if selection.ndim == 0:
            selection = selection.reshape(1)
        selection = wp.from_torch(selection)
    elif not isinstance(selection, wp.array):
        if device.is_capturing:
            raise RuntimeError("Mask resolution cannot upload host selectors during graph capture")
        temporary_input = True
        values = list(selection)
        if all(isinstance(value, (bool, np.bool_)) for value in values) and values:
            selection = wp.array(values, dtype=wp.bool, device=device)
        else:
            if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) for value in values):
                raise TypeError("Index sequences must contain integers")
            selection = wp.array(values, dtype=wp.int64, device=device)
    elif selection.device != device:
        if device.is_capturing:
            raise RuntimeError("Mask resolution cannot transfer selectors during graph capture")
        selection = selection.to(device)
        temporary_input = True

    if selection.ndim != 1:
        raise ValueError("Selectors must be rank-1 arrays")
    if mask is not None or selection.dtype == wp.bool:
        if selection.dtype != wp.bool or selection.shape != scratch_mask.shape:
            raise ValueError("Boolean masks must match the destination shape and dtype")
        return selection
    if selection.dtype not in (wp.int32, wp.int64):
        raise TypeError("Index arrays must use signed int32 or int64 values")
    scratch_mask.zero_()
    if selection.size:
        wp.launch(
            _make_populate_mask(selector_dtype(selection)), selection.shape, [scratch_mask, selection], device=device
        )
        if temporary_input and device.is_cuda:
            # Eager uploads must outlive scatter even on imported nonblocking CUDA streams.
            wp.synchronize_stream(wp.get_stream(device))
    return scratch_mask
