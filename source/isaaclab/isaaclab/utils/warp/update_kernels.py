# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import warp as wp


@wp.kernel
def update_array1D_with_value(
    new_value: Any,
    array: Any,
):
    """
    Assigns a value to all the elements of the array.

    Args:
        new_value: The new value.

    Modifies:
        array: The array to update.
    """
    index = wp.tid()
    array[index] = new_value


@wp.kernel
def update_array1D_with_value_masked(
    new_value: Any,
    array: Any,
    mask: wp.array(dtype=wp.bool),
):
    """
    Assigns a value to all the elements of the array where the mask is true.

    ..note:: if None is provided for mask, then all the elements are updated.

    Args:
        new_value: The new value.
        mask: The mask to use. Shape is (N,).

    Modifies:
        array: The array to update. Shape is (N,).
    """
    index = wp.tid()
    if not mask:  # noqa: SIM114
        array[index] = new_value
    elif mask[index]:
        array[index] = new_value


@wp.kernel
def update_array1D_with_value_indexed(
    new_value: Any,
    array: Any,
    indices: wp.array(dtype=wp.int32),
):
    """
    Assigns a value to all the elements of the array where the indices are true.

    Args:
        new_value: The new value.
        indices: The indices to use. Shape is (N,).

    Modifies:
        array: The array to update. Shape is (N,).
    """
    index = wp.tid()
    if not indices:
        array[index] = new_value
    else:
        array[indices[index]] = new_value


@wp.kernel
def update_array2D_with_value(
    new_value: Any,
    array_2d: Any,
):
    """
    Assigns a value to all the elements of the batched array.

    Args:
        new_value: The new value.

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    array_2d[index_1, index_2] = new_value


@wp.kernel
def update_array2D_with_value_masked(
    new_value: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    mask_2: wp.array(dtype=wp.bool),
):
    """
    Assigns a value to all the elements of the batched array where the masks are true.

    ..note:: if None is provided for mask_1 or mask_2, then all the rows or columns are updated respectively.

    Args:
        new_value: The new value.
        mask_1: The mask to use. Shape is (N,).
        mask_2: The mask to use. Shape is (M,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if not mask_1:  # noqa: SIM114
        if not mask_2:  # noqa: SIM114
            array_2d[index_1, index_2] = new_value
        elif mask_2[index_2]:
            array_2d[index_1, index_2] = new_value
    elif mask_1[index_1]:
        if not mask_2:  # noqa: SIM114
            array_2d[index_1, index_2] = new_value
        elif mask_2[index_2]:
            array_2d[index_1, index_2] = new_value


@wp.kernel
def update_array2D_with_value_indexed(
    new_value: Any,
    array_2d: Any,
    indices_1: wp.array(dtype=wp.int32),
    indices_2: wp.array(dtype=wp.int32),
):
    """
    Assigns a value to all the elements of the batched array where the indices are true.

    Args:
        new_value: The new value.
        indices_1: The indices to use. Shape is (N,).
        indices_2: The indices to use. Shape is (M,).
    """
    index_1, index_2 = wp.tid()
    if not indices_1 and not indices_2:
        array_2d[index_1, index_2] = new_value
    elif not indices_1:
        array_2d[index_1, indices_2[index_2]] = new_value
    elif not indices_2:
        array_2d[indices_1[index_1], index_2] = new_value
    else:
        array_2d[indices_1[index_1], indices_2[index_2]] = new_value


@wp.kernel
def update_array1D_with_array1D(
    new_array: Any,
    array: Any,
):
    """
    Assigns the elements of the new array to the elements of the array.

    Args:
        new_array: The new array. Shape is (N,).

    Modifies:
        array: The array to update. Shape is (N,).
    """
    index = wp.tid()
    array[index] = new_array[index]


@wp.kernel
def update_array1D_with_array1D_masked(
    new_array: Any,
    array: Any,
    mask: wp.array(dtype=wp.bool),
):
    """
    Assigns the elements of the new array to the elements of the array where the mask is true.

    ..note:: if None is provided for mask, then all the elements are updated.

    Args:
        new_array: The new array. Shape is (N,).
        mask: The mask to use. Shape is (N,).

    Modifies:
        array: The array to update. Shape is (N,).
    """
    index = wp.tid()
    if not mask:  # noqa: SIM114
        array[index] = new_array[index]
    elif mask[index]:
        array[index] = new_array[index]


@wp.kernel
def update_array1D_with_array1D_indexed(
    new_array: Any,
    array: Any,
    indices: wp.array(dtype=wp.int32),
):
    """
    Assigns the elements of the new array to the elements of the array where the indices are true.
    """
    index = wp.tid()
    if not indices:
        array[index] = new_array[index]
    else:
        array[indices[index]] = new_array[index]


@wp.kernel
def update_array2D_with_array1D(
    new_array: Any,
    array_2d: Any,
):
    """
    Assigns the elements of the new array to the elements of the batched array.

    Args:
        new_array: The new array. Shape is (M,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    array_2d[index_1, index_2] = new_array[index_2]


@wp.kernel
def update_array2D_with_array1D_masked(
    new_array: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    mask_2: wp.array(dtype=wp.bool),
):
    """
    Assigns the elements of the new array to the elements of the batched array where the masks are true.

    ..note:: if None is provided for mask_1 or mask_2, then all the rows or columns are updated respectively.

    Args:
        new_array: The new array. Shape is (M,).
        mask_1: The mask to use. Shape is (N,).
        mask_2: The mask to use. Shape is (M,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if not mask_1:  # noqa: SIM114
        if not mask_2:  # noqa: SIM114
            array_2d[index_1, index_2] = new_array[index_2]
        elif mask_2[index_2]:
            array_2d[index_1, index_2] = new_array[index_2]
    elif mask_1[index_1]:
        if not mask_2:  # noqa: SIM114
            array_2d[index_1, index_2] = new_array[index_2]
        elif mask_2[index_2]:
            array_2d[index_1, index_2] = new_array[index_2]


@wp.kernel
def update_array2D_with_array1D_indexed(
    new_array: Any,
    array_2d: Any,
    indices_1: wp.array(dtype=wp.int32),
    indices_2: wp.array(dtype=wp.int32),
):
    """
    Assigns the elements of the new array to the elements of the batched array where the indices are true.

    ..note:: M >= L and kernel launch dim should be (N, L)

    Args:
        new_array: The new array. Shape is (M,).
        array_2d: The array to update. Shape is (K, L).
        indices_1: The indices to use. Shape is (K,).
        indices_2: The indices to use. Shape is (L,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if not indices_1 and not indices_2:
        array_2d[index_1, index_2] = new_array[index_2]
    elif not indices_1:
        array_2d[index_1, indices_2[index_2]] = new_array[index_2]
    elif not indices_2:
        array_2d[indices_1[index_1], index_2] = new_array[index_2]
    else:
        array_2d[indices_1[index_1], indices_2[index_2]] = new_array[index_2]


@wp.kernel
def update_array2D_with_array1D_hybrid(
    new_array_1d: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    indices_2: wp.array(dtype=wp.int32),
):
    """
    Assigns the elements of the new array to the elements of the 2d array.
    The mask is used to determine the rows to update and the indices are used to determine the columns to update.

    ..note:: M >= L and the kernel launch dim should be (N, L)

    Args:
        new_array_1d: The new array. Shape is (M,).
        array_2d: The array to update. Shape is (N, L).
        mask_1: The mask to use. Shape is (N,).
        indices_2: The indices to use. Shape is (L,).

    Modifies:
        array_2d: The array to update. Shape is (N, L).
    """
    index_1, index_2 = wp.tid()
    if mask_1[index_1]:
        array_2d[index_1, index_2] = new_array_1d[indices_2[index_1]]


@wp.kernel
def update_array2D_with_array2D(
    new_array_2d: Any,
    array_2d: Any,
):
    """
    Assigns the elements of the new array to the elements of the batched array.

    Args:
        new_array_2d: The new array. Shape is (N, M).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    array_2d[index_1, index_2] = new_array_2d[index_1, index_2]


@wp.kernel
def update_array2D_with_array2D_masked(
    new_array_2d: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    mask_2: wp.array(dtype=wp.bool),
):
    """
    Assigns the elements of the new array to the elements of the batched array where the masks are true.

    ..note:: if None is provided for mask_1 or mask_2, then all the rows or columns are updated respectively.

    Args:
        new_array_2d: The new array. Shape is (N, M).
        array_2d: The array to update. Shape is (N, M).
        mask_1: The mask to use. Shape is (N,).
        mask_2: The mask to use. Shape is (M,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if not mask_1:  # noqa: SIM114
        if not mask_2:  # noqa: SIM114
            array_2d[index_1, index_2] = new_array_2d[index_1, index_2]
        elif mask_2[index_2]:
            array_2d[index_1, index_2] = new_array_2d[index_1, index_2]
    elif mask_1[index_1]:
        if not mask_2:  # noqa: SIM114
            array_2d[index_1, index_2] = new_array_2d[index_1, index_2]
        elif mask_2[index_2]:
            array_2d[index_1, index_2] = new_array_2d[index_1, index_2]


@wp.kernel
def update_array2D_with_array2D_indexed(
    new_array_2d: Any,
    array_2d: Any,
    indices_1: wp.array(dtype=wp.int32),
    indices_2: wp.array(dtype=wp.int32),
):
    """
    Assigns the elements of the new array to the elements of the 2d array where the indices are true.

    ..note:: N >= K and M >= L and the kernel launch dim should be (K, L)

    Args:
        new_array_2d: The new array. Shape is (N, M).
        array_2d: The array to update. Shape is (K, L).
        indices_1: The indices to use. Shape is (K,).
        indices_2: The indices to use. Shape is (L,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """

    index_1, index_2 = wp.tid()
    if not indices_1 and not indices_2:
        array_2d[index_1, index_2] = new_array_2d[index_1, index_2]
    elif not indices_1:
        array_2d[index_1, indices_2[index_2]] = new_array_2d[index_1, index_2]
    elif not indices_2:
        array_2d[indices_1[index_1], index_2] = new_array_2d[index_1, index_2]
    else:
        array_2d[indices_1[index_1], indices_2[index_2]] = new_array_2d[index_1, index_2]


@wp.kernel
def update_array2D_with_array2D_hybrid(
    new_array_2d: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    indices_2: wp.array(dtype=wp.int32),
):
    """
    Assigns the elements of the new array to the elements of the 2d array.
    The mask is used to determine the rows to update and the indices are used to determine the columns to update.

    ..note:: M >= L and kernel launch dim should be (N, L)

    Args:
        new_array_2d: The new array. Shape is (N, M).
        array_2d: The array to update. Shape is (N, L).
        mask_1: The mask to use. Shape is (N,).
        indices_2: The indices to use. Shape is (L,).

    Modifies:
        array_2d: The array to update. Shape is (N, L).
    """
    index_1, index_2 = wp.tid()
    if mask_1[index_1]:
        array_2d[index_1, index_2] = new_array_2d[index_1, indices_2[index_2]]


for dtype in [wp.float32, wp.int32, wp.bool, wp.vec2f, wp.vec3f, wp.quatf, wp.transformf, wp.spatial_vectorf]:
    wp.overload(update_array1D_with_value, {"new_value": dtype, "array": wp.array(dtype=dtype)})
    wp.overload(update_array1D_with_value_masked, {"new_value": dtype, "array": wp.array(dtype=dtype)})
    wp.overload(update_array1D_with_value_indexed, {"new_value": dtype, "array": wp.array(dtype=dtype)})
    wp.overload(update_array2D_with_value, {"new_value": dtype, "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_array2D_with_value_masked, {"new_value": dtype, "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_array2D_with_value_indexed, {"new_value": dtype, "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_array1D_with_array1D, {"new_array": wp.array(dtype=dtype), "array": wp.array(dtype=dtype)})
    wp.overload(
        update_array1D_with_array1D_masked, {"new_array": wp.array(dtype=dtype), "array": wp.array(dtype=dtype)}
    )
    wp.overload(
        update_array1D_with_array1D_indexed, {"new_array": wp.array(dtype=dtype), "array": wp.array(dtype=dtype)}
    )
    wp.overload(update_array2D_with_array1D, {"new_array": wp.array(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(
        update_array2D_with_array1D_masked, {"new_array": wp.array(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)}
    )
    wp.overload(
        update_array2D_with_array1D_indexed, {"new_array": wp.array(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)}
    )
    wp.overload(
        update_array2D_with_array2D, {"new_array_2d": wp.array2d(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)}
    )
    wp.overload(
        update_array2D_with_array2D_masked,
        {"new_array_2d": wp.array2d(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)},
    )
    wp.overload(
        update_array2D_with_array2D_indexed,
        {"new_array_2d": wp.array2d(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)},
    )
