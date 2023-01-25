# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for working with different array backends."""

import numpy as np
import torch
from typing import Optional, Sequence, Union

import warp as wp

__all__ = ["TENSOR_TYPES", "TENSOR_TYPE_CONVERSIONS", "convert_to_torch"]

TENSOR_TYPES = {
    "numpy": np.ndarray,
    "torch": torch.Tensor,
    "warp": wp.array,
}
"""A dictionary containing the types for each backend.

The keys are the name of the backend ("numpy", "torch", "warp") and the values are the corresponding type
(``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""

TENSOR_TYPE_CONVERSIONS = {
    "numpy": {wp.array: lambda x: x.numpy(), torch.Tensor: lambda x: x.detach().cpu().numpy()},
    "torch": {wp.array: lambda x: wp.torch.to_torch(x), np.ndarray: lambda x: torch.from_numpy(x)},
    "warp": {np.array: lambda x: wp.array(x), torch.Tensor: lambda x: wp.torch.from_torch(x)},
}
"""A nested dictionary containing the conversion functions for each backend.

The keys of the outer dictionary are the name of target backend ("numpy", "torch", "warp"). The keys of the
inner dictionary are the source backend (``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""


def convert_to_torch(
    array: Sequence[float],
    dtype: torch.dtype = None,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is :obj:`None`, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Args:
        array (Sequence[float]): The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype (torch.dtype, optional): Target data-type for the tensor.
        device (Optional[Union[torch.device, str]], optional): The target device for the tensor. Defaults to None.

    Returns:
        torch.Tensor: The converted array as torch tensor.
    """
    # Convert array to tensor
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    elif isinstance(array, wp.array):
        tensor = wp.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor
