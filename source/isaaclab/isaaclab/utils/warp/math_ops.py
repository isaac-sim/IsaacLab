# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from functools import lru_cache
from typing import Any

import warp as wp


def get_ndims(x: Any):
    if isinstance(x, wp.array):
        return x.ndim
    else:
        return 0


@lru_cache(maxsize=None)
def _get_clip_kernel(source_ndims: int, lower_ndims: int, upper_ndims: int):
    if source_ndims == 1:

        @wp.kernel
        def clip(x: wp.array(dtype=Any), lower: Any, upper: Any):
            i = wp.tid()
            x[i] = wp.clamp(
                x[i], wp.static(array_switch(lower_ndims))(lower, i), wp.static(array_switch(upper_ndims))(upper, i)
            )

    elif source_ndims == 2:

        @wp.kernel
        def clip(x: wp.array2d(dtype=Any), lower: Any, upper: Any):
            i, j = wp.tid()
            x[i, j] = wp.clamp(
                x[i, j],
                wp.static(array_switch(lower_ndims))(lower, i, j),
                wp.static(array_switch(upper_ndims))(upper, i, j),
            )

    elif source_ndims == 3:

        @wp.kernel
        def clip(x: wp.array3d(dtype=Any), lower: Any, upper: Any):
            i, j, k = wp.tid()
            x[i, j, k] = wp.clamp(
                x[i, j, k],
                wp.static(array_switch(lower_ndims))(lower, i, j, k),
                wp.static(array_switch(upper_ndims))(upper, i, j, k),
            )

    elif source_ndims == 4:

        @wp.kernel
        def clip(x: wp.array4d(dtype=Any), lower: Any, upper: Any):
            i, j, k, L = wp.tid()
            x[i, j, k, L] = wp.clamp(
                x[i, j, k, L],
                wp.static(array_switch(lower_ndims))(lower, i, j, k, L),
                wp.static(array_switch(upper_ndims))(upper, i, j, k, L),
            )

    return clip


@lru_cache(maxsize=None)
def _get_add_kernel(source_ndims: int, factor_ndims: int):
    if source_ndims == 1:

        @wp.kernel
        def add(x: wp.array(dtype=Any), factor: Any):
            i = wp.tid()
            x[i] = x[i] + wp.static(array_switch(factor_ndims))(factor, i)

    elif source_ndims == 2:

        @wp.kernel
        def add(x: wp.array2d(dtype=Any), factor: Any):
            i, j = wp.tid()
            x[i, j] = x[i, j] + wp.static(array_switch(factor_ndims))(factor, i, j)

    elif source_ndims == 3:

        @wp.kernel
        def add(x: wp.array3d(dtype=Any), factor: Any):
            i, j, k = wp.tid()
            x[i, j, k] = x[i, j, k] + wp.static(array_switch(factor_ndims))(factor, i, j, k)

    elif source_ndims == 4:

        @wp.kernel
        def add(x: wp.array4d(dtype=Any), factor: Any):
            i, j, k, L = wp.tid()
            x[i, j, k, L] = x[i, j, k, L] + wp.static(array_switch(factor_ndims))(factor, i, j, k, L)

    return add


@lru_cache(maxsize=None)
def _get_sub_kernel(source_ndims: int, factor_ndims: int):
    if source_ndims == 1:

        @wp.kernel
        def sub(x: wp.array(dtype=Any), factor: Any):
            i = wp.tid()
            x[i] = x[i] - wp.static(array_switch(factor_ndims))(factor, i)

    elif source_ndims == 2:

        @wp.kernel
        def sub(x: wp.array2d(dtype=Any), factor: Any):
            i, j = wp.tid()
            x[i, j] = x[i, j] - wp.static(array_switch(factor_ndims))(factor, i, j)

    elif source_ndims == 3:

        @wp.kernel
        def sub(x: wp.array3d(dtype=Any), factor: Any):
            i, j, k = wp.tid()
            x[i, j, k] = x[i, j, k] - wp.static(array_switch(factor_ndims))(factor, i, j, k)

    elif source_ndims == 4:

        @wp.kernel
        def sub(x: wp.array4d(dtype=Any), factor: Any):
            i, j, k, L = wp.tid()
            x[i, j, k, L] = x[i, j, k, L] - wp.static(array_switch(factor_ndims))(factor, i, j, k, L)

    return sub


@lru_cache(maxsize=None)
def _get_mul_kernel(source_ndims: int, factor_ndims: int):
    if source_ndims == 1:

        @wp.kernel
        def mul(x: wp.array(dtype=Any), factor: Any):
            i = wp.tid()
            x[i] = x[i] * wp.static(array_switch(factor_ndims))(factor, i)

    elif source_ndims == 2:

        @wp.kernel
        def mul(x: wp.array2d(dtype=Any), factor: Any):
            i, j = wp.tid()
            x[i, j] = x[i, j] * wp.static(array_switch(factor_ndims))(factor, i, j)

    elif source_ndims == 3:

        @wp.kernel
        def mul(x: wp.array3d(dtype=Any), factor: Any):
            i, j, k = wp.tid()
            x[i, j, k] = x[i, j, k] * wp.static(array_switch(factor_ndims))(factor, i, j, k)

    elif source_ndims == 4:

        @wp.kernel
        def mul(x: wp.array4d(dtype=Any), factor: Any):
            i, j, k, L = wp.tid()
            x[i, j, k, L] = x[i, j, k, L] * wp.static(array_switch(factor_ndims))(factor, i, j, k, L)

    return mul


@lru_cache(maxsize=None)
def _get_div_kernel(source_ndims: int, factor_ndims: int):
    if source_ndims == 1:

        @wp.kernel
        def div(x: wp.array(dtype=Any), factor: Any):
            i = wp.tid()
            x[i] = x[i] / wp.static(array_switch(factor_ndims))(factor, i)

    elif source_ndims == 2:

        @wp.kernel
        def div(x: wp.array2d(dtype=Any), factor: Any):
            i, j = wp.tid()
            x[i, j] = x[i, j] / wp.static(array_switch(factor_ndims))(factor, i, j)

    elif source_ndims == 3:

        @wp.kernel
        def div(x: wp.array3d(dtype=Any), factor: Any):
            i, j, k = wp.tid()
            x[i, j, k] = x[i, j, k] / wp.static(array_switch(factor_ndims))(factor, i, j, k)

    elif source_ndims == 4:

        @wp.kernel
        def div(x: wp.array4d(dtype=Any), factor: Any):
            i, j, k, L = wp.tid()
            x[i, j, k, L] = x[i, j, k, L] / wp.static(array_switch(factor_ndims))(factor, i, j, k, L)

    return div


@lru_cache(maxsize=None)
def _get_square_kernel(source_ndims: int):
    if source_ndims == 1:

        @wp.kernel
        def square(x: wp.array(dtype=Any)):
            i = wp.tid()
            x[i] = x[i] * x[i]

    elif source_ndims == 2:

        @wp.kernel
        def square(x: wp.array2d(dtype=Any)):
            i, j = wp.tid()
            x[i, j] = x[i, j] * x[i, j]

    elif source_ndims == 3:

        @wp.kernel
        def square(x: wp.array3d(dtype=Any)):
            i, j, k = wp.tid()
            x[i, j, k] = x[i, j, k] * x[i, j, k]

    elif source_ndims == 4:

        @wp.kernel
        def square(x: wp.array4d(dtype=Any)):
            i, j, k, L = wp.tid()
            x[i, j, k, L] = x[i, j, k, L] * x[i, j, k, L]

    return square


def array_switch(ndims: int):
    @wp.func
    def f_(values: Any, i: int = 0, j: int = 0, k: int = 0, L: int = 0) -> Any:
        if wp.static(ndims == 0):
            return values
        elif wp.static(ndims == 1):
            return values[i]
        elif wp.static(ndims == 2):
            return values[i, j]
        elif wp.static(ndims == 3):
            return values[i, j, k]
        elif wp.static(ndims == 4):
            return values[i, j, k, L]

    return f_


def inplace_clip(x: Any, lower: Any, upper: Any):
    # Pre-compute the number of dimensions for the lower and upper bounds so that we can hit the kernel cache
    lower_ndims = get_ndims(lower)
    upper_ndims = get_ndims(upper)
    source_ndims = get_ndims(x)
    # Fetch the appropriate kernel from the cache and launch it
    wp.launch(_get_clip_kernel(source_ndims, lower_ndims, upper_ndims), dim=x.shape, inputs=[x, lower, upper])


def inplace_add(x: Any, factor: Any):
    # Pre-compute the number of dimensions for the factor so that we can hit the kernel cache
    factor_ndims = get_ndims(factor)
    source_ndims = get_ndims(x)
    # Fetch the appropriate kernel from the cache and launch it
    wp.launch(_get_add_kernel(source_ndims, factor_ndims), dim=x.shape, inputs=[x, factor])


def inplace_sub(x: Any, factor: Any):
    # Pre-compute the number of dimensions for the factor so that we can hit the kernel cache
    factor_ndims = get_ndims(factor)
    source_ndims = get_ndims(x)
    # Fetch the appropriate kernel from the cache and launch it
    wp.launch(_get_sub_kernel(source_ndims, factor_ndims), dim=x.shape, inputs=[x, factor])


def inplace_mul(x: Any, factor: Any):
    # Pre-compute the number of dimensions for the factor so that we can hit the kernel cache
    factor_ndims = get_ndims(factor)
    source_ndims = get_ndims(x)
    # Fetch the appropriate kernel from the cache and launch it
    wp.launch(_get_mul_kernel(source_ndims, factor_ndims), dim=x.shape, inputs=[x, factor])


def inplace_div(x: Any, factor: Any):
    # Pre-compute the number of dimensions for the factor so that we can hit the kernel cache
    factor_ndims = get_ndims(factor)
    source_ndims = get_ndims(x)
    # Fetch the appropriate kernel from the cache and launch it
    wp.launch(_get_div_kernel(source_ndims, factor_ndims), dim=x.shape, inputs=[x, factor])


def inplace_square(x: Any):
    # Pre-compute the number of dimensions for the source so that we can hit the kernel cache
    source_ndims = get_ndims(x)
    # Fetch the appropriate kernel from the cache and launch it
    wp.launch(_get_square_kernel(source_ndims), dim=x.shape, inputs=[x])
