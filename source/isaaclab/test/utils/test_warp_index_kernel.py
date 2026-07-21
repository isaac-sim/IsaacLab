# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray, index_kernel
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


@pytest.mark.parametrize("selector_dtype", [wp.int32, wp.int64])
def test_dispatcher_selects_proxy_index_overloads_without_copy(selector_dtype: type) -> None:
    """Select and launch from the exact Warp allocations wrapped by proxy selectors."""
    env_array = wp.array([1, 0], dtype=selector_dtype, device="cpu")
    item_array = wp.array([2, 0], dtype=selector_dtype, device="cpu")
    env_ids = ProxyArray(env_array)
    item_ids = ProxyArray(item_array)

    kernel = _SCATTER_DISPATCHER.select(env_ids, item_ids)
    resolved_env_ids = index_kernel._selector_array(env_ids)
    resolved_item_ids = index_kernel._selector_array(item_ids)

    assert kernel is _SCATTER_DISPATCHER.select_dtypes(selector_dtype, selector_dtype)
    assert resolved_env_ids is env_array
    assert resolved_item_ids is item_array
    assert env_ids._torch_cache is None  # noqa: SLF001
    assert item_ids._torch_cache is None  # noqa: SLF001

    output = wp.zeros((2, 3), dtype=wp.int32, device="cpu")
    wp.launch(
        kernel,
        dim=(2, 2),
        inputs=[resolved_env_ids, resolved_item_ids],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), np.asarray([[1, 0, 1], [1, 0, 1]], dtype=np.int32))


@pytest.mark.parametrize(
    ("env_values", "item_values", "launch_dim", "expected"),
    [
        pytest.param([1, 0], [2, 0], (2, 2), [[1, 0, 1], [1, 0, 1]], id="noncontiguous-logical"),
        pytest.param([], [2, 0], (0, 2), [[0, 0, 0], [0, 0, 0]], id="empty"),
    ],
)
def test_dispatcher_preserves_proxy_logical_selections(
    env_values: list[int], item_values: list[int], launch_dim: tuple[int, int], expected: list[list[int]]
) -> None:
    """Preserve empty and noncontiguous logical selection behavior for proxies."""
    env_ids = ProxyArray(wp.array(env_values, dtype=wp.int32, device="cpu"))
    item_ids = ProxyArray(wp.array(item_values, dtype=wp.int32, device="cpu"))
    output = wp.zeros((2, 3), dtype=wp.int32, device="cpu")
    wp.launch(
        _SCATTER_DISPATCHER.select(env_ids, item_ids),
        dim=launch_dim,
        inputs=[index_kernel._selector_array(env_ids), index_kernel._selector_array(item_ids)],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), np.asarray(expected, dtype=np.int32))


def test_dispatcher_rejects_unsupported_index_dtype() -> None:
    env_ids = torch.tensor([0], dtype=torch.int16)
    item_ids = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit"):
        _SCATTER_DISPATCHER.select(env_ids, item_ids)


def test_dispatcher_rejects_unsupported_proxy_index_dtype() -> None:
    """Reject an unsupported proxy selector with the shared selector error."""
    env_ids = ProxyArray(wp.array([0], dtype=wp.int16, device="cpu"))
    item_ids = ProxyArray(wp.array([0], dtype=wp.int32, device="cpu"))
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit"):
        _SCATTER_DISPATCHER.select(env_ids, item_ids)


def test_dispatcher_selects_explicit_index_dtypes() -> None:
    """Select registered specializations without constructing selector arrays."""
    output = wp.zeros((2, 3), dtype=wp.int32, device="cpu")
    env_ids = wp.array([1, 0], dtype=wp.int64, device="cpu")
    item_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")
    wp.launch(
        _SCATTER_DISPATCHER.select_dtypes(wp.int64, wp.int32),
        dim=(2, 2),
        inputs=[env_ids, item_ids],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), np.asarray([[1, 0, 1], [1, 0, 1]], dtype=np.int32))


def test_dispatcher_rejects_unsupported_explicit_index_dtype() -> None:
    """Reject unsupported explicit selector dtypes before a Warp launch."""
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit"):
        _SCATTER_DISPATCHER.select_dtypes(wp.int16, wp.int32)


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
