# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for writing through masks with :func:`write_2d_with_mask`."""

import pytest
import torch
import warp as wp

from isaaclab.test.utils import test_devices
from isaaclab.utils.warp.utils import write_2d_with_mask


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("data_type", ["scalar", "torch", "warp", "torch_strided_vec2f"])
def test_write_2d_with_mask(device, data_type):
    """Only entries selected by both masks are written, and a short mask writes only the rows it covers.

    The environment mask and the data are views whose memory continues with a selected third row, so a launch over the
    full output would write that row too.
    """
    torch_device = wp.device_to_torch(device)
    env_mask = wp.array([True, False, True], dtype=wp.bool, device=device)[:2]
    item_mask = wp.array([False, True], dtype=wp.bool, device=device)
    if data_type == "torch_strided_vec2f":
        values = torch.arange(1, 13, dtype=torch.float32, device=torch_device).reshape(3, 2, 2)
        data = values.repeat_interleave(2, dim=-1)[..., ::2][:2]
        out = wp.zeros((3, 2), dtype=wp.vec2f, device=device)
    else:
        values = torch.arange(1, 7, dtype=torch.float32, device=torch_device).reshape(3, 2)
        data = {"scalar": 7.0, "torch": values[:2], "warp": wp.from_torch(values)[:2]}[data_type]
        out = wp.zeros((3, 2), dtype=wp.float32, device=device)
    write_2d_with_mask(data, env_mask, item_mask, out, device)
    expected = torch.zeros_like(values)
    expected[0, 1] = 7.0 if data_type == "scalar" else values[0, 1]
    assert torch.equal(wp.to_torch(out), expected)
