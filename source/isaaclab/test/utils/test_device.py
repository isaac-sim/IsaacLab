# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compute-device selection utilities."""

import contextlib
import sys
from types import SimpleNamespace

import isaaclab.utils.device as device_module
from isaaclab.utils.device import reserve_cuda_graph_memory, set_cuda_device


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


def test_reserve_cuda_graph_memory_captures_once_and_retries_after_failure(monkeypatch):
    """One allocation is recorded per device; a failed attempt is retried by the next call."""
    captures = []
    failures = [RuntimeError("capture failed")]
    device = SimpleNamespace(alias="cuda:0", is_cuda=True, is_mempool_enabled=True, is_capturing=False)

    @contextlib.contextmanager
    def scoped_capture(device, force_module_load):
        captures.append(device.alias)
        if failures:
            raise failures.pop()
        yield

    warp = SimpleNamespace(
        get_device=lambda alias: device, ScopedCapture=scoped_capture, empty=lambda *args, **kwargs: None, int32=None
    )
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setattr(device_module, "_GRAPH_MEMORY_RESERVED_DEVICES", set())

    for _ in range(3):
        reserve_cuda_graph_memory("cuda:0")

    assert captures == ["cuda:0", "cuda:0"]
