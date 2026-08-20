# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import product
from typing import Any

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
    """Dispatch a Warp kernel by its index-selector dtypes.

    Args:
        kernel: Generic Warp kernel to specialize.
        selector_names: Names of the kernel arguments that receive index selectors.
        argument_types: Concrete types for other generic kernel arguments. Defaults to None.
    """

    def __init__(
        self,
        kernel: wp.Kernel,
        selector_names: tuple[str, ...],
        argument_types: dict[str, Any] | None = None,
    ) -> None:
        if not selector_names:
            raise ValueError("selector_names must contain at least one kernel argument name.")
        argument_types = {} if argument_types is None else argument_types
        overlapping_names = set(selector_names).intersection(argument_types)
        if overlapping_names:
            names = ", ".join(sorted(overlapping_names))
            raise ValueError(f"argument_types must not redefine selector arguments: {names}.")
        self._selector_names = selector_names
        self._kernel = kernel
        self._overloads = {}
        for dtypes in product(_INDEX_DTYPES, repeat=len(selector_names)):
            signature = argument_types.copy()
            signature.update({name: wp.array(dtype=dtype) for name, dtype in zip(selector_names, dtypes, strict=True)})
            self._overloads[dtypes] = wp.overload(kernel, signature)

    def select_dtypes(self, *dtypes: type[wp.int32] | type[wp.int64]) -> wp.Kernel:
        """Select a specialization from explicit selector dtypes."""
        if len(dtypes) != len(self._selector_names):
            raise ValueError(
                f"Expected {len(self._selector_names)} selector dtypes for {self._selector_names}, got {len(dtypes)}."
            )
        for dtype in dtypes:
            if dtype not in _INDEX_DTYPES:
                raise TypeError(f"Index selector must use signed 32-bit or signed 64-bit integers, got {dtype}.")
        return self._overloads[dtypes]

    def select(self, *selectors: torch.Tensor | wp.array) -> wp.Kernel:
        """Select a specialization from Torch or Warp selectors."""
        if len(selectors) != len(self._selector_names):
            raise ValueError(
                f"Expected {len(self._selector_names)} selectors for {self._selector_names}, got {len(selectors)}."
            )
        dtypes = tuple(_selector_dtype(selector) for selector in selectors)
        return self.select_dtypes(*dtypes)
