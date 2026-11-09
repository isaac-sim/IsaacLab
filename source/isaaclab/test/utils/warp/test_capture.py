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

from isaaclab.utils.warp.capture import SYNC_DEBUG_ENV_VAR, GraphedCallable, GraphedCalls, eager, eager_method

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"


@wp.kernel
def _affine(values: wp.array(dtype=wp.int32), scale: wp.int32, shift: wp.int32):
    values[wp.tid()] = values[wp.tid()] * scale + shift


@wp.kernel
def _add(into: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.int32)):
    into[wp.tid()] = into[wp.tid()] + values[wp.tid()]


def _num_graphs(graphed: GraphedCallable) -> int:
    """Number of graphs recorded for a callable."""
    return sum(isinstance(step, wp.Graph) for steps, _ in graphed._recordings.values() for step in steps)


class _Affine:
    """Callable that applies ``x * scale + shift`` on the device and counts its Python invocations."""

    def __init__(self, scale: int = 1, shift: int = 1):
        self.scale, self.shift, self.calls = scale, shift, 0

    def __call__(self, values: wp.array, *args) -> wp.array:
        self.calls += 1
        wp.launch(_affine, dim=values.shape[0], inputs=[values, self.scale, self.shift], device=values.device)
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
    assert (_read(old), _read(new), fn.calls, _num_graphs(graphed)) == ([1], [3], 3, 3)


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


def test_calls_record_a_call_once_and_replay_it():
    fn = _Affine()
    calls = GraphedCalls(DEVICE, True)
    values = wp.zeros(2, dtype=wp.int32, device=DEVICE)

    calls(fn, values)
    assert calls(fn, values) is values
    assert (fn.calls, _read(values)) == (1, [2, 2])


def test_calls_copy_an_input_at_another_address_into_the_array_the_recording_reads():
    """A call passing a new array each time, like the actions of each step, replays one recording reading a copy."""
    total = wp.zeros(2, dtype=wp.int32, device=DEVICE)
    runs = []

    def add(values):
        runs.append(values)
        wp.launch(_add, dim=2, inputs=[total, values], device=DEVICE)

    calls = GraphedCalls(DEVICE, True)
    inputs = [wp.full(2, step + 1, dtype=wp.int32, device=DEVICE) for step in range(4)]
    for values in inputs:
        calls(add, values)
    assert _read(total) == [10, 10]
    assert len(runs) == 2, "the second array records once more, on a buffer that later calls copy into"
    assert _num_graphs(calls._graphed[add][0]) == 1


def test_block_records_its_calls_as_one_graph():
    first, second = _Affine(scale=2, shift=0), _Affine(scale=1, shift=3)
    calls = GraphedCalls(DEVICE, True)
    values = wp.ones(2, dtype=wp.int32, device=DEVICE)

    for _ in range(3):
        with calls.block("both"):
            assert calls(first, values) is values
            calls(second, values)
    assert _read(values) == [29, 29]
    assert (first.calls, second.calls) == (1, 1)
    steps = calls._blocks["both"]._steps
    assert len(steps) == 1 and isinstance(steps[0], wp.Graph)


def test_block_splits_its_graph_only_at_an_eager_call():
    first, second = _Affine(), _Affine()
    calls = GraphedCalls(DEVICE, True)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    host = []

    for _ in range(2):
        with calls.block("split"):
            calls(first, values)
            eager(host.append, 7)
            calls(second, values)
    assert [isinstance(step, wp.Graph) for step in calls._blocks["split"]._steps] == [True, False, True]
    assert host == [7, 7], "a replay repeats the eager call once, from the recording"
    assert _read(values) == [4]


def test_block_replay_answers_its_calls_from_the_recording_and_runs_at_its_end():
    fn = _Affine()
    calls = GraphedCalls(DEVICE, True)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with calls.block("b"):
        calls(fn, values)
    with calls.block("b"):
        assert calls(fn, values) is values
        assert fn.calls == 1, "a replay does not run the calls inside the block"
    assert _read(values) == [2]


def test_block_making_other_calls_than_recorded_raises():
    first, second = _Affine(), _Affine()
    calls = GraphedCalls(DEVICE, True)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with calls.block("b"):
        calls(first, values)
    with pytest.raises(RuntimeError, match="other calls"), calls.block("b"):
        calls(second, values)


def test_block_on_other_arrays_runs_eagerly_and_records_again():
    fn = _Affine()
    calls = GraphedCalls(DEVICE, True)
    old, new = wp.zeros(1, dtype=wp.int32, device=DEVICE), wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with calls.block("b"):
        calls(fn, old)
    for _ in range(3):
        with calls.block("b"):
            calls(fn, new)
    assert (_read(old), _read(new), fn.calls) == ([1], [3], 3)


def test_block_records_only_its_own_calls():
    """A call that a call of the block makes is part of that call, not a call of the block."""
    inner = _Affine()
    calls = GraphedCalls(DEVICE, True)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    def outer(values):
        return calls(inner, values)

    for _ in range(2):
        with calls.block("b"):
            calls(outer, values)
    assert [fn for fn, _, _ in calls._blocks["b"]._made] == [outer]
    assert (inner.calls, _read(values)) == (1, [2])


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


def test_calls_without_recording_are_plain_calls():
    fn = _Affine()
    calls = GraphedCalls(DEVICE, False)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with calls.block("b"):
        calls(fn, values)
    calls(fn, values)
    assert (fn.calls, _read(values), calls._graphed, calls._blocks) == (2, [2], {}, {})
