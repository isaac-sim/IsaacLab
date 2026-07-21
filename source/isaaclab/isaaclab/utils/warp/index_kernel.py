# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import product

import torch
import warp as wp

_INDEX_DTYPES = (wp.int32, wp.int64)


def _selector_dtype(selector: torch.Tensor | wp.array):
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
    """Register and select signed-integer specializations of one Warp kernel."""

    def __init__(self, kernel: wp.Kernel, selector_names: tuple[str, ...]):
        if not selector_names:
            raise ValueError("selector_names must contain at least one kernel argument name.")
        self._selector_names = selector_names
        self._overloads = {}
        for dtypes in product(_INDEX_DTYPES, repeat=len(selector_names)):
            signature = {name: wp.array(dtype=dtype) for name, dtype in zip(selector_names, dtypes, strict=True)}
            self._overloads[dtypes] = wp.overload(kernel, signature)

    def select(self, *selectors: torch.Tensor | wp.array) -> wp.Kernel:
        if len(selectors) != len(self._selector_names):
            raise ValueError(
                f"Expected {len(self._selector_names)} selectors for {self._selector_names}, got {len(selectors)}."
            )
        dtypes = tuple(_selector_dtype(selector) for selector in selectors)
        return self._overloads[dtypes]
