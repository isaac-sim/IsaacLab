# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Value-preserving operations on caller-supplied array layouts and selections.

Callers resolve their attribute frequency into array shapes, strides and selectors.
This module neither names frequencies nor interprets element components. Cached factories
return concrete kernels on demand. Warm up the execution path before graph capture.
"""

from collections.abc import Sequence
from functools import cache
from numbers import Integral
from typing import Any

import numpy as np
import torch
import warp as wp

from .index_kernel import selector_dtype


def _component_layout(dtypes: tuple) -> tuple[type, tuple[int, ...]]:
    """Validate one-dimensional composite elements and return their scalar type and sizes."""
    if not dtypes or any(len(getattr(dtype, "_shape_", ())) != 1 for dtype in dtypes):
        raise TypeError("Expected one-dimensional composite element types")
    scalar_dtype = dtypes[0]._wp_scalar_type_
    if any(dtype._wp_scalar_type_ != scalar_dtype for dtype in dtypes):
        raise TypeError("Component scalar types must match")
    return scalar_dtype, tuple(wp.types.type_size(dtype) for dtype in dtypes)


@cache
def make_concat(first_dtype, second_dtype, result_dtype, ndim: int) -> wp.Kernel:
    """Concatenate two typed elements in one launch, retaining each array's strides.

    Elements must have a one-dimensional component layout (no scalars or matrices).
    Inputs and output must have matching shapes, rank 1 or 2, and the same scalar
    type. The output component count must equal the sum of the inputs' counts.
    """
    if ndim not in (1, 2):
        raise ValueError(f"Expected rank 1 or 2, got {ndim}")
    _, (first_size, second_size, result_size) = _component_layout((first_dtype, second_dtype, result_dtype))
    if first_size + second_size != result_size:
        raise ValueError("Output component count must equal the sum of the input counts")

    @wp.kernel(module="unique")
    def concatenate(
        first: wp.array(dtype=first_dtype, ndim=ndim),
        second: wp.array(dtype=second_dtype, ndim=ndim),
        result: wp.array(dtype=result_dtype, ndim=ndim),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            a, b = first[i], second[i]
        else:
            i, j = wp.tid()
            a, b = first[i, j], second[i, j]
        value = result_dtype()
        for k in range(wp.static(first_size)):
            value[k] = a[k]
        for k in range(wp.static(second_size)):
            value[first_size + k] = b[k]
        if wp.static(ndim == 1):
            result[i] = value
        else:
            result[i, j] = value

    return concatenate


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


@wp.func
def _selected_index(selection: wp.array(dtype=Any), i: int) -> wp.int64:
    """Map a launch index through a signed integer selector."""
    return wp.int64(selection[i])


@wp.func
def _selected_index(selection: wp.array(dtype=wp.bool), i: int) -> wp.int64:
    """Keep a launch index unless an optional boolean mask excludes it."""
    if selection and not selection[i]:
        return wp.int64(-1)
    return wp.int64(i)


@cache
def make_copy(dtype, ndim: int, row_dtype=wp.bool, column_dtype=wp.bool, *, full_source: bool = False) -> wp.Kernel:
    """Copy selected elements, specializing only the requested layout.

    Boolean selectors mask the destination's first two axes; ``None`` selects
    the whole axis. Integer selectors use the Cartesian product of nonnegative
    row/column indices. Rank-1 arrays ignore columns. Remaining components of
    rank-3 arrays are copied unchanged. Array strides are preserved.

    With ``full_source=True``, read the source at destination indices; otherwise
    read at launch indices (a compact source for integer selections). The kernel
    takes ``(source, rows, columns, target, selected_rows)``. The optional int32
    ``selected_rows`` output records row indices once per selected row.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"Expected rank 1, 2 or 3, got {ndim}")
    if row_dtype not in (wp.bool, wp.int32, wp.int64) or column_dtype not in (wp.bool, wp.int32, wp.int64):
        raise TypeError("Selectors must use bool, int32 or int64")

    @wp.kernel(module="unique")
    def copy(
        source: wp.array(dtype=dtype, ndim=ndim),
        rows: wp.array(dtype=row_dtype),
        columns: wp.array(dtype=column_dtype),
        target: wp.array(dtype=dtype, ndim=ndim),
        selected_rows: wp.array(dtype=wp.int32),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            j, k = 0, 0
        elif wp.static(ndim == 2):
            i, j = wp.tid()
            k = 0
        else:
            i, j, k = wp.tid()
        row = _selected_index(rows, i)
        column = wp.int64(0)
        if wp.static(ndim > 1):
            column = _selected_index(columns, j)
        if row < 0 or column < 0:
            return
        if selected_rows and j == 0 and k == 0:
            selected_rows[i] = wp.int32(row)

        if wp.static(full_source):
            a, b = row, column
        else:
            a, b = wp.int64(i), wp.int64(j)
        if wp.static(ndim == 1):
            target[row] = source[a]
        elif wp.static(ndim == 2):
            target[row, column] = source[a, b]
        else:
            target[row, column, k] = source[a, b, k]

    return copy


@cache
def make_fill(dtype, ndim: int, row_dtype=wp.bool, column_dtype=wp.bool) -> wp.Kernel:
    """Fill selected elements with one value, using the selectors of :func:`make_copy`.

    The kernel takes ``(value, rows, columns, target, selected_rows)``. It has no
    source array or source-layout flag. Each specialization is an isolated kernel.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"Expected rank 1, 2 or 3, got {ndim}")
    if row_dtype not in (wp.bool, wp.int32, wp.int64) or column_dtype not in (wp.bool, wp.int32, wp.int64):
        raise TypeError("Selectors must use bool, int32 or int64")

    @wp.kernel(module="unique")
    def fill(
        value: dtype,
        rows: wp.array(dtype=row_dtype),
        columns: wp.array(dtype=column_dtype),
        target: wp.array(dtype=dtype, ndim=ndim),
        selected_rows: wp.array(dtype=wp.int32),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            j, k = 0, 0
        elif wp.static(ndim == 2):
            i, j = wp.tid()
            k = 0
        else:
            i, j, k = wp.tid()
        row = _selected_index(rows, i)
        column = wp.int64(0)
        if wp.static(ndim > 1):
            column = _selected_index(columns, j)
        if row < 0 or column < 0:
            return
        if selected_rows and j == 0 and k == 0:
            selected_rows[i] = wp.int32(row)
        if wp.static(ndim == 1):
            target[row] = value
        elif wp.static(ndim == 2):
            target[row, column] = value
        else:
            target[row, column, k] = value

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
