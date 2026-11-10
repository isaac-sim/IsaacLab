# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the CUDA graph recording of the Warp frontend's step parts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp.capture import SYNC_DEBUG_ENV_VAR, GraphedCallable, eager, eager_method

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"


@wp.kernel
def _affine(values: wp.array(dtype=wp.int32), scale: wp.int32, shift: wp.int32):
    values[wp.tid()] = values[wp.tid()] * scale + shift


def _num_graphs(graphed: GraphedCallable) -> int:
    """Number of graphs recorded for a callable."""
    return sum(isinstance(step, wp.Graph) for steps, _ in graphed._recordings.values() for step in steps)


class _Affine:
    """Callable that applies ``x * scale + shift`` on the device and counts its Python invocations.

    A ``shift`` passed to the call replaces the configured one for that call.
    """

    def __init__(self, scale: int = 1, shift: int = 1):
        self.scale, self.shift, self.calls = scale, shift, 0

    def __call__(self, values: wp.array, shift: int | None = None) -> wp.array:
        self.calls += 1
        shift = self.shift if shift is None else shift
        wp.launch(_affine, dim=values.shape[0], inputs=[values, self.scale, shift], device=values.device)
        return values


def _read(values: wp.array) -> list[int]:
    wp.synchronize_device(values.device)
    return values.numpy().tolist()


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.delenv(SYNC_DEBUG_ENV_VAR, raising=False)
    monkeypatch.setattr(GraphedCallable, "enabled", True)


def test_first_call_records_and_runs_the_callable_once():
    """Recording has no eager warm-up: a stateful callable advances once per call, including the first."""
    fn = _Affine()
    graphed = GraphedCallable(fn, DEVICE)
    values = wp.zeros(2, dtype=wp.int32, device=DEVICE)

    assert graphed(values) is values
    assert _read(values) == [1, 1]
    assert graphed(values) is values
    assert _read(values) == [2, 2]
    assert (fn.calls, _num_graphs(graphed)) == (1, 1), "the second call must replay the recorded graph"


def test_changed_argument_records_the_callable_again():
    """A graph replays its recorded pointers and scalars, so a new array or scalar records a new graph."""
    fn = _Affine()
    graphed = GraphedCallable(fn, DEVICE)
    old, new = wp.zeros(1, dtype=wp.int32, device=DEVICE), wp.zeros(1, dtype=wp.int32, device=DEVICE)

    graphed(old)
    graphed(new)
    graphed(new)
    graphed(new, 5)
    graphed(new, 5)
    # replays of the shift-5 recording add 5 each: 0 -> 1 -> 2 -> 7 -> 12
    assert (_read(old), _read(new), fn.calls, _num_graphs(graphed)) == ([1], [12], 3, 3)


def test_views_of_one_array_record_their_own_graphs():
    """Views that share an address but differ in length record their own graphs, each replaying over its length."""
    fn = _Affine()
    graphed = GraphedCallable(fn, DEVICE)
    values = wp.zeros(4, dtype=wp.int32, device=DEVICE)

    graphed(values[:2])
    graphed(values[:4])
    graphed(values[:4])
    assert (_read(values), fn.calls, _num_graphs(graphed)) == ([3, 3, 2, 2], 2, 2)


def test_physics_rebind_records_the_callable_again():
    fn = _Affine()
    graphed = GraphedCallable(fn, DEVICE)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    graphed(values)

    GraphedCallable.invalidate()
    graphed(values)
    graphed(values)
    assert (fn.calls, _read(values), _num_graphs(graphed)) == (2, [3], 1)


def test_callable_runs_eagerly_when_disabled_or_on_cpu(monkeypatch):
    on_cpu = _Affine()
    cpu_graphed = GraphedCallable(on_cpu, "cpu")
    cpu_values = wp.zeros(1, dtype=wp.int32, device="cpu")
    cpu_graphed(cpu_values)
    cpu_graphed(cpu_values)

    monkeypatch.setattr(GraphedCallable, "enabled", False)
    disabled = _Affine()
    graphed = GraphedCallable(disabled, DEVICE)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    graphed(values)
    graphed(values)

    assert (on_cpu.calls, _read(cpu_values), _num_graphs(cpu_graphed)) == (2, [2], 0)
    assert (disabled.calls, _read(values), _num_graphs(graphed)) == (2, [2], 0)


def test_eager_call_runs_between_its_graphs_on_every_call():
    """The recording runs the graph before an eager call first, and every replay keeps that order."""
    before, between, after = _Affine(2, 0), _Affine(1, 10), _Affine(3, 0)

    def body(values: wp.array) -> wp.array:
        before(values)
        eager(between, values)
        return after(values)

    graphed = GraphedCallable(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    graphed(values)
    # the non-commuting steps only give these values in order: 1 * 2 + 10 = 12 -> 36, then 36 * 2 + 10 = 82 -> 246
    assert _read(values) == [36]
    graphed(values)
    assert (_read(values), between.calls, before.calls, after.calls, _num_graphs(graphed)) == ([246], 2, 1, 1, 2)


def test_eager_call_first_runs_before_the_recorded_work():
    """A callable making its eager call first runs it before its recorded work, also when it replays the recording."""
    eager_fn, first, second = _Affine(1, 10), _Affine(2, 0), _Affine(3, 0)

    def body(values: wp.array) -> wp.array:
        eager(eager_fn, values)
        first(values)
        return second(values)

    graphed = GraphedCallable(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)
    graphed(values)
    graphed(values)

    assert _read(values) == [((1 + 10) * 6 + 10) * 6]


def test_eager_call_of_a_nested_callable_splits_the_enclosing_recording():
    eager_fn, inner_fn, outer_fn = _Affine(1, 10), _Affine(2, 0), _Affine(3, 0)
    nested = GraphedCallable(lambda values: (eager(eager_fn, values), inner_fn(values)), DEVICE)
    outer = GraphedCallable(lambda values: (outer_fn(values), nested(values)), DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    outer(values)
    outer(values)
    # 1 * 3 + 10 = 13 -> 26, then 26 * 3 + 10 = 88 -> 176: the nested eager call runs between the outer graphs
    assert (_read(values), eager_fn.calls, inner_fn.calls, outer_fn.calls) == ([176], 2, 1, 1)
    assert (_num_graphs(outer), _num_graphs(nested)) == (2, 0)


def test_eager_call_raises_inside_a_capture_no_callable_started():
    """A call that cannot be recorded must not be captured silently into a foreign graph."""
    fn = _Affine()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="no GraphedCallable started"):
        with wp.ScopedDevice(DEVICE), wp.ScopedCapture():
            eager(fn, values)
    assert fn.calls == 0


def test_callable_raising_while_recording_ends_the_capture():
    """A failed recording must not leave the device capturing; the next call records again."""
    recorded, eager_fn = _Affine(2, 0), _Affine(1, 10)

    def body(values: wp.array) -> wp.array:
        recorded(values)
        eager(eager_fn, values)
        if recorded.calls == 1:
            raise ValueError("first recording fails")
        return values

    graphed = GraphedCallable(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    with pytest.raises(ValueError, match="first recording fails"):
        graphed(values)
    assert not wp.get_device(DEVICE).is_capturing
    graphed(values)
    # the failed recording ran its first graph and the eager call: 1 * 2 + 10 = 12, then 12 * 2 + 10 = 34
    assert (_read(values), _num_graphs(graphed)) == ([34], 2)


def test_sync_debug_trap_raises_on_a_hidden_host_sync(monkeypatch):
    monkeypatch.setattr(GraphedCallable, "enabled", False)
    monkeypatch.setenv(SYNC_DEBUG_ENV_VAR, "1")
    flags = torch.ones(4, dtype=torch.bool, device=DEVICE)
    graphed = GraphedCallable(lambda: flags.any().item(), DEVICE)

    with pytest.raises(RuntimeError, match="synchroniz"):
        graphed()
    assert torch.cuda.get_sync_debug_mode() == 0, "the trap must be released after the call"


class _ChangeRegistry:
    """Host-side registry, like a solver's queue of model changes, that a recording cannot hold."""

    def __init__(self):
        self.changes = []

    @eager_method
    def add(self, change: int):
        self.changes.append(change)


def test_eager_method_repeats_host_work_on_every_replay():
    """A recorded writer's host registration runs on the recording and on each replay, after the recorded kernel."""
    registry = _ChangeRegistry()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    fn = _Affine()

    def writer():
        fn(values)
        registry.add(7)

    graphed = GraphedCallable(writer, DEVICE)
    for _ in range(3):
        graphed()
    assert (registry.changes, _read(values), fn.calls) == ([7, 7, 7], [3], 1)


def test_recording_settles_the_watched_backend_first():
    """Work the physics backend queued before a recording runs once, before it, not on every replay."""
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    class Backend:
        pending = True

        def register_callback(self, fn, event):
            return SimpleNamespace(deregister=lambda: None)

        def forward(self):
            if self.pending:
                wp.launch(_affine, dim=1, inputs=[values, 1, 1], device=DEVICE)
                self.pending = False

    backend = Backend()
    GraphedCallable.watch(backend)
    try:
        graphed = GraphedCallable(backend.forward, DEVICE)
        for _ in range(3):
            graphed()
    finally:
        GraphedCallable.unwatch()
    assert _read(values) == [1]
