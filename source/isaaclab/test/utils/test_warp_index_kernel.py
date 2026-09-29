# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher


@wp.kernel
def _scatter_indices(
    env_ids: wp.array(dtype=Any),
    item_ids: wp.array(dtype=Any),
    output: wp.array2d(dtype=wp.int32),
) -> None:
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    item_id = wp.int32(item_ids[j])
    output[env_id, item_id] = 1


@wp.kernel
def _scatter_values(
    env_ids: wp.array(dtype=Any),
    item_ids: wp.array(dtype=Any),
    values: wp.array2d(dtype=Any),
    output: wp.array2d(dtype=Any),
) -> None:
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    item_id = wp.int32(item_ids[j])
    output[env_id, item_id] = values[i, j]


_SCATTER_DISPATCHER = IndexKernelDispatcher(_scatter_indices, ("env_ids", "item_ids"))


@pytest.mark.parametrize("env_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("item_dtype", [torch.int32, torch.int64])
def test_dispatcher_selects_torch_index_overloads(env_dtype: torch.dtype, item_dtype: torch.dtype) -> None:
    env_ids = torch.tensor([1, 0], dtype=env_dtype)
    item_ids = torch.tensor([2, 0], dtype=item_dtype)
    output = wp.zeros((2, 3), dtype=wp.int32, device="cpu")
    wp.launch(
        _SCATTER_DISPATCHER.select(env_ids, item_ids),
        dim=(2, 2),
        inputs=[env_ids, item_ids],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), np.asarray([[1, 0, 1], [1, 0, 1]], dtype=np.int32))


def test_dispatcher_selects_torch_index_overload_without_creating_warp_view(monkeypatch) -> None:
    """Select a Torch overload from dtype metadata without constructing a Warp view."""
    from_torch_calls = 0
    original_from_torch = wp.from_torch

    def record_from_torch(*args, **kwargs):
        nonlocal from_torch_calls
        from_torch_calls += 1
        return original_from_torch(*args, **kwargs)

    monkeypatch.setattr(wp, "from_torch", record_from_torch)
    env_ids = torch.tensor([1, 0], dtype=torch.int64)
    item_ids = torch.tensor([2, 0], dtype=torch.int32)

    kernel = _SCATTER_DISPATCHER.select(env_ids, item_ids)

    assert kernel is _SCATTER_DISPATCHER.select_dtypes(wp.int64, wp.int32)
    assert from_torch_calls == 0


def test_dispatcher_requires_explicit_proxy_view() -> None:
    """Reject output wrappers until callers select their intended array representation."""
    env_ids = ProxyArray(wp.array([0], dtype=wp.int32, device="cpu"))
    item_ids = ProxyArray(wp.array([1], dtype=wp.int64, device="cpu"))

    with pytest.raises(TypeError, match="torch.Tensor or wp.array"):
        _SCATTER_DISPATCHER.select(env_ids, item_ids)

    _SCATTER_DISPATCHER.select(env_ids.warp, item_ids.warp)
    _SCATTER_DISPATCHER.select(env_ids.torch, item_ids.torch)


def test_dispatcher_rejects_unsupported_index_dtype() -> None:
    env_ids = torch.tensor([0], dtype=torch.int16)
    item_ids = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit"):
        _SCATTER_DISPATCHER.select(env_ids, item_ids)


def test_dispatcher_supports_fixed_non_selector_generic_types() -> None:
    dispatcher = IndexKernelDispatcher(
        _scatter_values,
        ("env_ids", "item_ids"),
        argument_types={
            "values": wp.array2d(dtype=wp.float32),
            "output": wp.array2d(dtype=wp.float32),
        },
    )
    env_ids = torch.tensor([1, 0], dtype=torch.int64)
    item_ids = torch.tensor([2, 0], dtype=torch.int32)
    values = wp.array(np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32), device="cpu")
    output = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    wp.launch(
        dispatcher.select(env_ids, item_ids),
        dim=(2, 2),
        inputs=[env_ids, item_ids, values],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), np.asarray([[21.0, 0.0, 20.0], [11.0, 0.0, 10.0]]))
