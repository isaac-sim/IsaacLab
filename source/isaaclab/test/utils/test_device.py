# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compute-device selection utilities."""

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch

import isaaclab.utils.device as device_module
from isaaclab.utils.device import reserve_cuda_graph_memory, set_cuda_device

pytestmark = pytest.mark.unit


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


def _fake_warp(monkeypatch, *, is_cuda=True, is_mempool_enabled=True, is_capturing=False, capture_error=None):
    """Install a fake ``warp`` module that records captures and allocations."""
    calls = []
    device = SimpleNamespace(
        alias="cuda:0" if is_cuda else "cpu",
        is_cuda=is_cuda,
        is_mempool_enabled=is_mempool_enabled,
        is_capturing=is_capturing,
    )

    @contextlib.contextmanager
    def scoped_capture(device, force_module_load):
        calls.append(("capture", device.alias, force_module_load))
        if capture_error is not None:
            raise capture_error
        yield

    warp = SimpleNamespace(
        get_device=lambda alias: device,
        ScopedCapture=scoped_capture,
        empty=lambda shape, dtype, device: calls.append(("empty", shape, device.alias)),
        int32="int32",
    )
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setattr(device_module, "_GRAPH_MEMORY_RESERVED_DEVICES", set())
    return calls


def test_reserve_cuda_graph_memory_captures_one_allocation_once(monkeypatch):
    """The first call records one allocation in a graph; later calls for the same device do nothing."""
    calls = _fake_warp(monkeypatch)

    reserve_cuda_graph_memory("cuda:0")
    reserve_cuda_graph_memory("cuda:0")

    assert calls == [("capture", "cuda:0", False), ("empty", 1, "cuda:0")]


@pytest.mark.parametrize(
    "device_state",
    [{"is_cuda": False}, {"is_mempool_enabled": False}, {"is_capturing": True}],
    ids=["cpu", "mempool_disabled", "already_capturing"],
)
def test_reserve_cuda_graph_memory_skips_unsupported_devices(monkeypatch, device_state):
    """Devices that cannot record an allocation in a graph are left alone."""
    calls = _fake_warp(monkeypatch, **device_state)

    reserve_cuda_graph_memory("cuda:0")

    assert calls == []
    assert not device_module._GRAPH_MEMORY_RESERVED_DEVICES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_reserve_cuda_graph_memory_on_cuda_keeps_later_captures_working(monkeypatch, caplog):
    """The reservation succeeds, and later captures can still allocate and run."""
    import warp as wp

    device = wp.get_device("cuda:0")
    if not device.is_mempool_enabled:
        pytest.skip("Warp memory pool is disabled")
    monkeypatch.setattr(device_module, "_GRAPH_MEMORY_RESERVED_DEVICES", set())

    with caplog.at_level("WARNING", logger=device_module.__name__):
        reserve_cuda_graph_memory(device.alias)

    assert "Could not reserve" not in caplog.text
    assert device.alias in device_module._GRAPH_MEMORY_RESERVED_DEVICES
    assert not device.is_capturing
    output = wp.zeros(4, dtype=wp.int32, device=device)
    with wp.ScopedCapture(device=device) as capture:
        scratch = wp.full(4, 5, dtype=wp.int32, device=device)
        wp.copy(output, scratch)
        del scratch
    wp.capture_launch(capture.graph)
    assert output.numpy().tolist() == [5, 5, 5, 5]


def test_reserve_cuda_graph_memory_logs_capture_failure(monkeypatch, caplog):
    """A failed reservation must not prevent startup."""
    _fake_warp(monkeypatch, capture_error=RuntimeError("capture failed"))

    with caplog.at_level("WARNING", logger=device_module.__name__):
        reserve_cuda_graph_memory("cuda:0")

    assert "Could not reserve CUDA graph memory on cuda:0" in caplog.text
