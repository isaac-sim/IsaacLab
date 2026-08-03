# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compute-device selection utilities."""

import sys
from types import SimpleNamespace

from isaaclab.utils._device import set_cuda_device


def test_set_cuda_device_sets_torch_before_warp(monkeypatch):
    """Setting a CUDA device must update PyTorch before Warp."""
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(set_device=lambda device: calls.append(("torch", device))))
    warp = SimpleNamespace(set_device=lambda device: calls.append(("warp", device)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "warp", warp)

    set_cuda_device("cuda:2")

    assert calls == [("torch", "cuda:2"), ("warp", "cuda:2")]


def test_set_cuda_device_normalizes_integer_for_warp(monkeypatch):
    """An integer device index must be converted to a Warp CUDA alias."""
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(set_device=lambda device: calls.append(("torch", device))))
    warp = SimpleNamespace(set_device=lambda device: calls.append(("warp", device)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "warp", warp)

    set_cuda_device(3)

    assert calls == [("torch", 3), ("warp", "cuda:3")]
