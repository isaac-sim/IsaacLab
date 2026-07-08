# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the CapturedKernelUpdate CUDA-graph helper."""

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import CapturedKernelUpdate
from isaaclab.utils.warp import capture as capture_module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

DEVICE = "cuda:0"


def _make_copy_compute():
    source = wp.ones(1, dtype=wp.int32, device=DEVICE)
    destination = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    calls = {"count": 0}

    def compute() -> None:
        calls["count"] += 1
        wp.copy(destination, source)

    return source, destination, calls, compute


def test_run_captures_once_and_replays_fresh_data():
    """The first run should capture and execute; later runs replay without re-running Python."""
    source, destination, calls, compute = _make_copy_compute()
    graph = CapturedKernelUpdate(DEVICE, owner="test sensor at '/World/S'")

    graph.run(compute)
    wp.synchronize_device(DEVICE)
    assert calls["count"] == 1
    assert destination.numpy().tolist() == [1]

    source.fill_(2)
    graph.run(compute)
    wp.synchronize_device(DEVICE)
    assert calls["count"] == 1
    assert destination.numpy().tolist() == [2]


def test_run_falls_back_eagerly_when_capture_fails(monkeypatch):
    """A capture failure should disable graphs permanently and still produce data."""
    _, destination, calls, compute = _make_copy_compute()
    graph = CapturedKernelUpdate(DEVICE, owner="test sensor at '/World/S'")

    class _FailingCapture:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("capture unavailable")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(capture_module.wp, "ScopedCapture", _FailingCapture)
    graph.run(compute)
    wp.synchronize_device(DEVICE)
    assert graph.enabled is False
    assert calls["count"] == 1
    assert destination.numpy().tolist() == [1]

    graph.run(compute)
    wp.synchronize_device(DEVICE)
    assert calls["count"] == 2


def test_refuse_outer_capture_raises_inside_capture():
    """Refusal must fire while an outer capture is active, and not otherwise."""
    source, destination, _, _ = _make_copy_compute()
    graph = CapturedKernelUpdate(DEVICE, owner="test sensor at '/World/S'")

    graph.refuse_outer_capture()  # no active capture: no raise

    with wp.ScopedCapture(device=wp.get_device(DEVICE)):
        with pytest.raises(RuntimeError, match="CUDA graph capture is active"):
            graph.refuse_outer_capture()
        # record something so the outer capture does not end empty
        wp.copy(destination, source)


def test_invalidate_forces_recapture():
    """After invalidate, the next run must capture (and execute) again."""
    _, _, calls, compute = _make_copy_compute()
    graph = CapturedKernelUpdate(DEVICE, owner="test sensor at '/World/S'")

    graph.run(compute)
    graph.invalidate()
    graph.run(compute)
    wp.synchronize_device(DEVICE)
    assert calls["count"] == 2


def test_is_captured_tracks_graph_lifecycle():
    """``is_captured`` is False before the first run, True after, and False after invalidate."""
    _, _, _, compute = _make_copy_compute()
    graph = CapturedKernelUpdate(DEVICE, owner="test sensor at '/World/S'")

    assert graph.is_captured is False
    graph.run(compute)
    wp.synchronize_device(DEVICE)
    assert graph.is_captured is True
    graph.invalidate()
    assert graph.is_captured is False


def test_cpu_device_stays_eager():
    """On CPU the helper must be disabled and run compute eagerly every call."""
    calls = {"count": 0}

    def compute() -> None:
        calls["count"] += 1

    graph = CapturedKernelUpdate("cpu", owner="test sensor at '/World/S'")
    assert graph.enabled is False
    graph.run(compute)
    graph.run(compute)
    assert calls["count"] == 2
