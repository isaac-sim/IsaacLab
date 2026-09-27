# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for working with different array backends."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import torch
import warp as wp

from .warp.proxy_array import ProxyArray

TensorData = Union[np.ndarray, torch.Tensor, wp.array]  # noqa: UP007
"""Type definition for a tensor data.

Union of numpy, torch, and warp arrays.
"""

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
    "warp": {np.ndarray: lambda x: wp.array(x), torch.Tensor: lambda x: wp.torch.from_torch(x)},
}
"""A nested dictionary containing the conversion functions for each backend.

The keys of the outer dictionary are the name of target backend ("numpy", "torch", "warp"). The keys of the
inner dictionary are the source backend (``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""


def index_fill_(
    data: torch.Tensor, indices: Sequence[int] | torch.Tensor | slice | None, value: float | int | bool, dim: int = 0
) -> None:
    """Fill selected entries in place without synchronizing device-resident integer indices.

    Slices fill a view; integer indices use :meth:`torch.Tensor.index_fill_`. Assigning a Python
    scalar through integer tensor indexing instead uploads the scalar and synchronizes CUDA.
    Host indices still require an upload; int32 indices require a device-side cast to int64.

    Args:
        data: Tensor to modify, including non-contiguous views.
        indices: Integer indices, a one-dimensional boolean mask, or a slice along ``dim``.
            None fills the entire array.
        value: Scalar value to write.
        dim: Dimension selected by ``indices``. Defaults to zero.
    """
    if indices is None:
        data.fill_(value)
    elif isinstance(indices, slice):
        selection = [slice(None)] * data.ndim
        selection[dim] = indices
        data[tuple(selection)].fill_(value)
    else:
        indices = torch.as_tensor(indices, device=data.device)
        if indices.dtype == torch.bool:
            shape = [1] * data.ndim
            shape[dim] = -1
            data.masked_fill_(indices.reshape(shape), value)
        else:
            data.index_fill_(dim, indices.to(dtype=torch.long), value)


def torch_index(indices: Sequence[int] | slice | ProxyArray) -> Sequence[int] | slice | torch.Tensor:
    """Return a Torch index for an index selection, such as a resolved scene-entity selection.

    Finalized selections are :class:`~isaaclab.utils.warp.ProxyArray` objects; their cached device tensor is
    returned so indexing does not upload or synchronize. Slices and host sequences are returned unchanged.

    Args:
        indices: Index selection.

    Returns:
        The selection in a form accepted by Torch indexing and asset index arguments.
    """
    return indices.torch if isinstance(indices, ProxyArray) else indices


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
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    elif isinstance(array, wp.array):
        if array.dtype == wp.uint32:
            array = array.view(wp.int32)
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
