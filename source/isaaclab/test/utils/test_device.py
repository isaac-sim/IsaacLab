# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compute-device selection utilities."""

import pytest
import torch
import warp as wp

from isaaclab.utils._device import set_cuda_device

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("device", "warp_device"), [("cuda:2", "cuda:2"), (3, "cuda:3")])
def test_set_cuda_device_sets_torch_before_warp(monkeypatch, device, warp_device):
    """Select PyTorch's device before Warp, normalizing integer indices for Warp."""
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("torch", device)))
    monkeypatch.setattr(wp, "set_device", lambda device: calls.append(("warp", device)))

    set_cuda_device(device)

    assert calls == [("torch", device), ("warp", warp_device)]
