# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-time generation of stride-preserving elementwise kernels."""

import warp as wp


def make_binary(operation, left_dtype, right_dtype, result_dtype, ndim: int, *, module: str) -> wp.Kernel:
    """Map a binary device operation over equally shaped rank-1 or rank-2 arrays.

    Construct the required concrete kernels at module import, before launch or capture.
    Arrays retain their own strides; inputs and output must have matching shapes.
    Pass the owner's __name__ as module so importing another owner cannot invalidate
    this owner's compiled kernels by adding variants to the factory module.
    """
    if ndim not in (1, 2):
        raise ValueError(f"Expected rank 1 or 2, got {ndim}")

    @wp.kernel(module=module)
    def apply(
        left: wp.array(dtype=left_dtype, ndim=ndim),
        right: wp.array(dtype=right_dtype, ndim=ndim),
        result: wp.array(dtype=result_dtype, ndim=ndim),
    ):
        if wp.static(ndim == 1):
            i = wp.tid()
            result[i] = operation(left[i], right[i])
        else:
            i, j = wp.tid()
            result[i, j] = operation(left[i, j], right[i, j])

    return apply
