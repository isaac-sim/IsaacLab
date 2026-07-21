# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import product

import torch
import warp as wp

_INDEX_DTYPES = (wp.int32, wp.int64)


def _selector_dtype(selector: torch.Tensor | wp.array) -> type[wp.int32] | type[wp.int64]:
    if isinstance(selector, torch.Tensor):
        dtype = wp.dtype_from_torch(selector.dtype)
    elif isinstance(selector, wp.array):
        dtype = selector.dtype
    else:
        raise TypeError(f"Index selector must be a torch.Tensor or wp.array, got {type(selector).__name__}.")
    if dtype not in _INDEX_DTYPES:
        raise TypeError(f"Index selector must use signed 32-bit or signed 64-bit integers, got {dtype}.")
    return dtype


class IndexKernelDispatcher:
    """Register and select signed-integer specializations of one Warp kernel.

    Args:
        kernel: Generic Warp kernel to specialize.
        selector_names: Names of the kernel arguments that receive index selectors.

    Raises:
        ValueError: If :paramref:`selector_names` is empty.
    """

    def __init__(self, kernel: wp.Kernel, selector_names: tuple[str, ...]) -> None:
        if not selector_names:
            raise ValueError("selector_names must contain at least one kernel argument name.")
        self._selector_names = selector_names
        self._overloads = {}
        for dtypes in product(_INDEX_DTYPES, repeat=len(selector_names)):
            signature = {name: wp.array(dtype=dtype) for name, dtype in zip(selector_names, dtypes, strict=True)}
            self._overloads[dtypes] = wp.overload(kernel, signature)

    def select(self, *selectors: torch.Tensor | wp.array) -> wp.Kernel:
        """Select the specialization matching the index selector dtypes.

        Args:
            selectors: Torch tensors or Warp arrays with signed 32-bit or 64-bit integer dtypes.

        Returns:
            The concrete Warp kernel specialized for the selector dtypes.

        Raises:
            TypeError: If a selector is not a Torch tensor or Warp array with a supported integer dtype.
            ValueError: If the number of selectors does not match the number of selector argument names.
        """
        if len(selectors) != len(self._selector_names):
            raise ValueError(
                f"Expected {len(self._selector_names)} selectors for {self._selector_names}, got {len(selectors)}."
            )
        dtypes = tuple(_selector_dtype(selector) for selector in selectors)
        return self._overloads[dtypes]
