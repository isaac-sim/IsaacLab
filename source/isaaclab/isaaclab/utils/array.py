# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for working with different array backends."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import warp as wp

    TensorData = np.ndarray | torch.Tensor | wp.array
else:
    TensorData = Any
"""Type definition for a tensor data.

Union of numpy, torch, and warp arrays.
"""


def _import_warp():
    import warp as wp

    return wp


class _LazyTensorTypes(dict):
    """Mapping of backend names to tensor types with Warp resolved on demand."""

    _WARP_KEY = "warp"

    def __init__(self):
        super().__init__({"numpy": np.ndarray, "torch": torch.Tensor})

    def _ensure_warp(self) -> None:
        if not dict.__contains__(self, self._WARP_KEY):
            super().__setitem__(self._WARP_KEY, _import_warp().array)

    def __contains__(self, key: object) -> bool:
        return key == self._WARP_KEY or super().__contains__(key)

    def __getitem__(self, key: str):
        if key == self._WARP_KEY:
            self._ensure_warp()
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key == self._WARP_KEY:
            self._ensure_warp()
        return super().get(key, default)

    def __iter__(self):
        self._ensure_warp()
        return super().__iter__()

    def __len__(self):
        self._ensure_warp()
        return super().__len__()

    def keys(self):
        self._ensure_warp()
        return super().keys()

    def items(self):
        self._ensure_warp()
        return super().items()

    def values(self):
        self._ensure_warp()
        return super().values()


class _LazyTensorTypeConversions(dict):
    """Mapping of backend names to conversion functions with Warp resolved on demand."""

    _SUPPORTED_KEYS = ("numpy", "torch", "warp")

    def _ensure_conversions(self) -> None:
        if super().__len__() == len(self._SUPPORTED_KEYS):
            return
        wp = _import_warp()
        super().update(
            {
                "numpy": {wp.array: lambda x: x.numpy(), torch.Tensor: lambda x: x.detach().cpu().numpy()},
                "torch": {wp.array: lambda x: wp.torch.to_torch(x), np.ndarray: lambda x: torch.from_numpy(x)},
                "warp": {np.array: lambda x: wp.array(x), torch.Tensor: lambda x: wp.torch.from_torch(x)},
            }
        )

    def __contains__(self, key: object) -> bool:
        return key in self._SUPPORTED_KEYS

    def __getitem__(self, key: str):
        self._ensure_conversions()
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key in self._SUPPORTED_KEYS:
            self._ensure_conversions()
        return super().get(key, default)

    def __iter__(self):
        self._ensure_conversions()
        return super().__iter__()

    def __len__(self):
        self._ensure_conversions()
        return super().__len__()

    def keys(self):
        self._ensure_conversions()
        return super().keys()

    def items(self):
        self._ensure_conversions()
        return super().items()

    def values(self):
        self._ensure_conversions()
        return super().values()


TENSOR_TYPES = _LazyTensorTypes()
"""A dictionary containing the types for each backend.

The keys are the name of the backend ("numpy", "torch", "warp") and the values are the corresponding type
(``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""

TENSOR_TYPE_CONVERSIONS = _LazyTensorTypeConversions()
"""A nested dictionary containing the conversion functions for each backend.

The keys of the outer dictionary are the name of target backend ("numpy", "torch", "warp"). The keys of the
inner dictionary are the source backend (``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    warp_module = sys.modules.get("warp")

    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    elif warp_module is not None and isinstance(array, warp_module.array):
        if array.dtype == warp_module.uint32:
            array = array.view(warp_module.int32)
        tensor = warp_module.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor
