# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.array import index_fill_
from isaaclab.utils.dict import convert_dict_to_backend

pytestmark = pytest.mark.unit


def test_convert_numpy_array_to_warp_backend():
    data = {"values": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    converted = convert_dict_to_backend(data, backend="warp", array_types=("numpy",))

    assert isinstance(converted["values"], wp.array)
    np.testing.assert_array_equal(converted["values"].numpy(), data["values"])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("dtype,value", [(torch.float32, 2.5), (torch.int64, -7), (torch.bool, True)])
@pytest.mark.parametrize(
    "indices", [None, slice(None), slice(1, None, 2), slice(0, 0), [], [2, 0, 2, -1], "mask", "scalar"]
)
def test_index_fill(device, dim, dtype, value, indices):
    """Slices and integer selections update only their entries, including strided views, without CUDA sync."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data = torch.zeros((5, 8, 4), dtype=dtype, device=device).transpose(0, 1)
    expected = data.cpu().clone()
    if indices == "mask":
        indices = torch.arange(data.shape[dim]) % 2 == 0
    elif indices == "scalar":
        indices = torch.tensor(2)
    selection = [slice(None)] * data.ndim
    if indices is not None:
        selection[dim] = indices
    expected[tuple(selection)] = value
    if isinstance(indices, list) and device.startswith("cuda"):
        indices = torch.tensor(indices, dtype=torch.int64 if dtype == torch.int64 else torch.int32, device=device)
    elif isinstance(indices, torch.Tensor):
        indices = indices.to(device)
    previous = torch.cuda.get_sync_debug_mode() if device.startswith("cuda") else None
    if previous is not None:
        torch.cuda.set_sync_debug_mode("error")
    try:
        index_fill_(data, indices, value, dim)
    finally:
        if previous is not None:
            torch.cuda.set_sync_debug_mode(previous)
    torch.testing.assert_close(data.cpu(), expected)
