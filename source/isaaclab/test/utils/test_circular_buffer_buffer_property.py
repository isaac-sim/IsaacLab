# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils import CircularBuffer

pytestmark = pytest.mark.unit


def test_buffer_property_rejects_access_before_first_append():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")

    with pytest.raises(RuntimeError, match="append data"):
        _ = buffer.buffer


def test_buffer_property_preserves_post_append_behavior():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    buffer.append(data)

    expected = data.unsqueeze(1).repeat(1, buffer.max_length, 1)
    torch.testing.assert_close(buffer.buffer, expected)
