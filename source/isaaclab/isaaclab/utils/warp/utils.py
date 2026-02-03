# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence

import warp as wp

logger = logging.getLogger(__name__)

##
# Frontend conversions - Torch to Warp.
##


def make_complete_data_from_torch_single_index(
    value: torch.Tensor,
    N: int,
    ids: Sequence[int] | torch.Tensor | None = None,
    dtype: type = wp.float32,
    device: str = "cuda:0",
    out: wp.array | None = None,
) -> wp.array:
    """Converts any Torch frontend data into warp data with single index support.

    Args:
        value: The value to convert. Shape is (N,) or (len(ids),).
        N: The number of elements in the complete array.
        ids: The index ids. If None, value is expected to be complete data.
            For best performance, pass a torch.Tensor instead of a Python list.
        dtype: The dtype of the value.
        device: The device to use for the conversion.
        out: Optional pre-allocated warp array to write into. If provided, avoids memory
            allocation. The array will be zeroed before writing. Shape must be (N, *value.shape[1:]).

    Returns:
        A warp array. If `out` is provided, returns `out`.
    """
    # Treat slice(None) as equivalent to None (means "select all")
    # Reject other slice types as they're not supported
    if isinstance(ids, slice):
        if ids == slice(None):
            ids = None
        else:
            raise ValueError(f"Only slice(None) is supported for ids, got {ids}")

    if ids is None:
        # No ids are provided, so we are expecting complete data.
        if out is not None:
            # Copy into pre-allocated buffer
            out_torch = wp.to_torch(out)
            out_torch.copy_(value)
            return out
        else:
            return wp.from_torch(value, dtype=dtype)
    else:
        # Convert list to tensor once at the start (list->cuda tensor is expensive)
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long, device=device)

        if out is not None:
            # Use pre-allocated buffer - zero it and fill with indexed values
            complete = wp.to_torch(out)
            complete.zero_()
            complete[ids] = value
            return out
        else:
            # Create a complete data buffer from scratch
            complete = torch.zeros((N, *value.shape[1:]), dtype=torch.float32, device=device)
            complete[ids] = value
            return wp.from_torch(complete, dtype=dtype)


def make_complete_data_from_torch_dual_index(
    value: torch.Tensor,
    N: int,
    M: int,
    first_ids: Sequence[int] | torch.Tensor | None = None,
    second_ids: Sequence[int] | torch.Tensor | None = None,
    dtype: type = wp.float32,
    device: str = "cuda:0",
    out: wp.array | None = None,
) -> wp.array:
    """Converts any Torch frontend data into warp data with dual index support.

    Args:
        value: The value to convert. Shape is (N, M) or (len(first_ids), len(second_ids)).
        N: The number of elements in the first dimension.
        M: The number of elements in the second dimension.
        first_ids: The first index ids.
            For best performance, pass a torch.Tensor instead of a Python list.
        second_ids: The second index ids.
            For best performance, pass a torch.Tensor instead of a Python list.
        dtype: The dtype of the value.
        device: The device to use for the conversion.
        out: Optional pre-allocated warp array to write into. If provided, avoids memory
            allocation. The array will be zeroed before writing. Shape must be (N, M, *value.shape[2:]).

    Returns:
        A warp array. If `out` is provided, returns `out`.
    """
    if (first_ids is None) and (second_ids is None):
        # No ids are provided, so we are expecting complete data.
        if out is not None:
            # Copy into pre-allocated buffer
            out_torch = wp.to_torch(out)
            out_torch.copy_(value)
            return out
        else:
            return wp.from_torch(value, dtype=dtype)
    else:
        # Get or create the complete data buffer
        if out is not None:
            complete = wp.to_torch(out)
            complete.zero_()
        else:
            complete = torch.zeros((N, M, *value.shape[2:]), dtype=torch.float32, device=device)

        # Convert lists to tensors once at the start (list->cuda tensor is expensive)
        # Treat slice(None) as equivalent to None (means "select all")
        # Reject other slice types as they're not supported
        if isinstance(first_ids, slice):
            if first_ids == slice(None):
                first_ids = None
            else:
                raise ValueError(f"Only slice(None) is supported for first_ids, got {first_ids}")
        elif first_ids is not None and not isinstance(first_ids, torch.Tensor):
            first_ids = torch.tensor(first_ids, dtype=torch.long, device=device)

        if isinstance(second_ids, slice):
            if second_ids == slice(None):
                second_ids = None
            else:
                raise ValueError(f"Only slice(None) is supported for second_ids, got {second_ids}")
        elif second_ids is not None and not isinstance(second_ids, torch.Tensor):
            second_ids = torch.tensor(second_ids, dtype=torch.long, device=device)

        # Fill the complete data buffer with the value.
        # For dual indexing with both tensors, need to reshape for broadcasting
        if first_ids is not None and second_ids is not None:
            complete[first_ids[:, None], second_ids] = value
        elif first_ids is not None:
            complete[first_ids, :] = value
        elif second_ids is not None:
            complete[:, second_ids] = value
        else:
            # Both are None/slice - copy entire value
            complete[:] = value

        if out is not None:
            return out
        else:
            return wp.from_torch(complete, dtype=dtype)


def make_mask_from_torch_ids(
    N: int,
    ids: Sequence[int] | torch.Tensor | None = None,
    mask: wp.array | torch.Tensor | None = None,
    device: str = "cuda:0",
    out: wp.array | None = None,
) -> wp.array | None:
    """Converts any Torch frontend ids into warp mask.

    Args:
        N: The number of elements in the array.
        ids: The index ids.
            For best performance, pass a torch.Tensor instead of a Python list.
        mask: The index mask. If provided as warp array, returned directly.
        device: The device to use for the conversion.
        out: Optional pre-allocated warp bool array to write into. If provided, avoids memory
            allocation. The array will be zeroed before writing. Shape must be (N,).

    Returns:
        A warp mask. None if no ids and no mask are provided.
    """
    # Convert list to tensor once at the start (list->cuda tensor is expensive)
    # Treat slice(None) as equivalent to None (means "select all")
    # Reject other slice types as they're not supported
    if isinstance(ids, slice):
        if ids == slice(None):
            ids = None
        else:
            raise ValueError(f"Only slice(None) is supported for ids, got {ids}")
    elif ids is not None and not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long, device=device)

    if (ids is not None) and (mask is None):
        if out is not None:
            # Use pre-allocated buffer - zero it and set indexed values
            mask_torch = wp.to_torch(out)
            mask_torch.zero_()
            mask_torch[ids] = True
            return out
        else:
            # Create a mask from scratch
            mask_torch = torch.zeros(N, dtype=torch.bool, device=device)
            mask_torch[ids] = True
            return wp.from_torch(mask_torch, dtype=wp.bool)
    elif isinstance(mask, torch.Tensor):
        if out is not None:
            out_torch = wp.to_torch(out)
            out_torch.copy_(mask)
            return out
        else:
            return wp.from_torch(mask, dtype=wp.bool)
    return mask
