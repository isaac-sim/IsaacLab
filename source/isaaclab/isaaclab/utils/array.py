# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for working with different array backends."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, Union

import numpy as np
import torch
import warp as wp

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


def env_mask_from_ids(
    env_ids: Sequence[int] | torch.Tensor | slice | None, num_envs: int, device: torch.device | str
) -> torch.Tensor:
    """Return a boolean mask that selects ``env_ids`` without synchronizing the device.

    Args:
        env_ids: Integer indices, a boolean mask, or a slice. None selects every environment.
        num_envs: Number of environments.
        device: Device of the mask.

    Returns:
        Boolean mask. Shape is (num_envs,).
    """
    if isinstance(env_ids, torch.Tensor) and env_ids.dtype == torch.bool:
        return env_ids
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    index_fill_(mask, slice(None) if env_ids is None else env_ids, True)
    return mask


def env_ids_from_mask(env_mask: torch.Tensor | None) -> torch.Tensor | slice:
    """Return the indices selected by ``env_mask``.

    This synchronizes the device, since the number of indices is data dependent. Use it only for consumers that
    take indices.

    Args:
        env_mask: Boolean mask. None selects every environment.

    Returns:
        Long indices on the mask's device, or ``slice(None)`` when ``env_mask`` is None.
    """
    if env_mask is None:
        return slice(None)
    return env_mask.nonzero().squeeze(-1)


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


def env_selection_kwargs(
    fn: Callable[..., Any],
    env_mask: torch.Tensor,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
) -> dict[str, Any] | None:
    """Return the keyword argument that selects environments for ``fn``.

    Callables that declare an ``env_mask`` parameter receive the boolean mask and must leave unselected
    environments unchanged. They run on every reset, so the environment step never waits on the device. Other
    callables receive indices as ``env_ids``: the caller's ``env_ids`` when given, else the indices selected by
    ``env_mask``. Computing those synchronizes the device, and the call is skipped when none are selected.

    Args:
        fn: Term function, term instance, or term method.
        env_mask: Boolean mask of the selected environments. Shape is (num_envs,).
        env_ids: The indices or slice the caller selected ``env_mask`` with, if any. Defaults to None.

    Returns:
        ``{"env_mask": env_mask}`` or ``{"env_ids": indices}``, or None when ``fn`` takes indices and none are
        selected.
    """
    if takes_env_mask(fn):
        return {"env_mask": env_mask}
    if env_ids is None:
        env_ids = env_ids_from_mask(env_mask)
        if len(env_ids) == 0:
            return None
    return {"env_ids": env_ids}


def takes_env_mask(fn: Callable[..., Any]) -> bool:
    """Whether ``fn`` selects environments with an ``env_mask`` parameter.

    Args:
        fn: Function, class, callable instance, bound method, or :func:`functools.partial`.
    """
    if isinstance(fn, functools.partial):
        return "env_mask" in inspect.signature(fn).parameters
    return _declares_env_mask(_signature_target(fn))


def _signature_target(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return the plain function whose signature describes calling ``fn``; cached by identity."""
    if inspect.ismethod(fn):
        return fn.__func__
    if inspect.isfunction(fn):
        return fn
    return (fn if inspect.isclass(fn) else type(fn)).__call__


@functools.cache
def _declares_env_mask(fn: Callable[..., Any]) -> bool:
    return "env_mask" in inspect.signature(fn).parameters
