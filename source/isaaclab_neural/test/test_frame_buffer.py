# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent camera readers of model frames, including masked resets and unchanged steps."""

import pytest
import torch

from isaaclab.test.utils import test_devices

from isaaclab_neural.physics import FrameBuffer


@pytest.mark.parametrize("device", test_devices())
def test_readers_have_independent_progress_and_preserve_unselected_environments(device):
    buffer = FrameBuffer(3, 4, (2, 2), device)
    first, second = buffer.reader(), buffer.reader()
    history_a, history_b = torch.zeros_like(buffer.frames), torch.zeros_like(buffer.frames)
    mask_a = torch.zeros(3, dtype=torch.bool, device=device)
    mask_b = torch.zeros_like(mask_a)
    chunk = torch.full_like(buffer.frames, 7)
    start = torch.full((3, 1, 2, 2, 3), 9, dtype=torch.uint8, device=device)
    selected = torch.tensor([False, True, False], device=device)
    on_cuda = device.startswith("cuda")
    previous_mode = torch.cuda.get_sync_debug_mode() if on_cuda else None
    if on_cuda:
        torch.cuda.set_sync_debug_mode("error")
    try:
        buffer.publish(chunk)
        assert first.read_into(history_a, mask_a)
        assert not first.read_into(history_a, mask_a)
        assert second.read_into(history_b, mask_b)
    finally:
        if on_cuda:
            torch.cuda.set_sync_debug_mode(previous_mode)
    assert not mask_a.any() and mask_b.all()
    assert (history_a == 7).all() and (history_b == 7).all()

    buffer.publish(start, selected)
    assert first.read_into(history_a, mask_a) and second.read_into(history_b, mask_b)
    assert mask_a.tolist() == mask_b.tolist() == [False, True, False]
    assert history_a[:, :, 0, 0, 0].tolist() == history_b[:, :, 0, 0, 0].tolist() == [[7] * 4, [9] * 4, [7] * 4]
