# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import pytest
import torch
import warp as wp

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


def test_dispatcher_rejects_unsupported_index_dtype() -> None:
    env_ids = torch.tensor([0], dtype=torch.int16)
    item_ids = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit"):
        _SCATTER_DISPATCHER.select(env_ids, item_ids)
