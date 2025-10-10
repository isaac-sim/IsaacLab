import warp as wp
from typing import Any

@wp.kernel
def update_array_with_value(
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
def update_array_with_value_masked(
    new_value: Any,
    array: Any,
    mask: wp.array(dtype=wp.bool),
):
    """
    Assigns a value to all the elements of the array where the mask is true.
    
    Args:
        new_value: The new value.
        mask: The mask to use. Shape is (N,).
    
    Modifies:
        array: The array to update. Shape is (N,).
    """
    index = wp.tid()
    if mask[index]:
        array[index] = new_value


@wp.kernel
def update_batched_array_with_value(
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
def update_batched_array_with_value_masked(
    new_value: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    mask_2: wp.array(dtype=wp.bool),
):
    """
    Assigns a value to all the elements of the batched array where the masks are true.

    Args:
        new_value: The new value.
        mask_1: The mask to use. Shape is (N,).
        mask_2: The mask to use. Shape is (M,).
    
    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if mask_1[index_1] and mask_2[index_2]:
        array_2d[index_1, index_2] = new_value

@wp.kernel
def update_array_with_array(
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
def update_array_with_array_masked(
    new_array: Any,
    array: Any,
    mask: wp.array(dtype=wp.bool),
):
    """
    Assigns the elements of the new array to the elements of the array where the mask is true.

    Args:
        new_array: The new array. Shape is (N,).
        mask: The mask to use. Shape is (N,).

    Modifies:
        array: The array to update. Shape is (N,).
    """
    index = wp.tid()
    if mask[index]:
        array[index] = new_array[index]

@wp.kernel
def update_batched_array_with_array(
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
def update_batched_array_with_array_masked(
    new_array: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    mask_2: wp.array(dtype=wp.bool),
):
    """
    Assigns the elements of the new array to the elements of the batched array where the masks are true.

    Args:
        new_array: The new array. Shape is (M,).
        mask_1: The mask to use. Shape is (N,).
        mask_2: The mask to use. Shape is (M,).

    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if mask_1[index_1] and mask_2[index_2]:
        array_2d[index_1, index_2] = new_array[index_2]

@wp.kernel
def update_batched_array_with_batched_array(
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
def update_batched_array_with_batched_array_masked(
    new_array_2d: Any,
    array_2d: Any,
    mask_1: wp.array(dtype=wp.bool),
    mask_2: wp.array(dtype=wp.bool),
):
    """
    Assigns the elements of the new array to the elements of the batched array where the masks are true.

    Args:
        new_array_2d: The new array. Shape is (N, M).
        mask_1: The mask to use. Shape is (N,).
        mask_2: The mask to use. Shape is (M,).
    
    Modifies:
        array_2d: The array to update. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    if mask_1[index_1] and mask_2[index_2]:
        array_2d[index_1, index_2] = new_array_2d[index_1, index_2]


for dtype in [wp.float32, wp.int32, wp.bool, wp.vec2f, wp.vec3f, wp.quatf, wp.transformf, wp.spatial_vectorf]:
    wp.overload(update_array_with_value, {"new_value": dtype, "array": wp.array(dtype=dtype)})
    wp.overload(update_array_with_value_masked, {"new_value": dtype, "array": wp.array(dtype=dtype)})
    wp.overload(update_batched_array_with_value, {"new_value": dtype, "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_batched_array_with_value_masked, {"new_value": dtype, "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_array_with_array, {"new_array": wp.array(dtype=dtype), "array": wp.array(dtype=dtype)})
    wp.overload(update_array_with_array_masked, {"new_array": wp.array(dtype=dtype), "array": wp.array(dtype=dtype)})
    wp.overload(update_batched_array_with_array, {"new_array": wp.array(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_batched_array_with_array_masked, {"new_array": wp.array(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_batched_array_with_batched_array, {"new_array_2d": wp.array2d(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)})
    wp.overload(update_batched_array_with_batched_array_masked, {"new_array_2d": wp.array2d(dtype=dtype), "array_2d": wp.array2d(dtype=dtype)})
    