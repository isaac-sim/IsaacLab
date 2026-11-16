# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the CUDA graph recording of the Warp frontend's step parts."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp.capture import (
    GraphedCallable,
    capturing,
    eager,
    eager_method,
)

wp.init()
pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")]

DEVICE = "cuda:0"


@wp.kernel
def _affine(values: wp.array(dtype=wp.int32), scale: wp.int32, shift: wp.int32):
    values[wp.tid()] = values[wp.tid()] * scale + shift


def _num_graphs(graphed: GraphedCallable) -> int:
    """Number of graphs recorded for a callable."""
    return len(graphed._recordings)


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


def test_equal_arguments_of_another_layout_or_type_record_their_own_graphs():
    """Arguments that compare equal but differ in position, keyword or scalar type each record a graph."""
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    calls = []

    def fn(*args, **kwargs):
        calls.append((args, kwargs))
        wp.launch(_affine, dim=1, inputs=[values, 1, 1], device=DEVICE)

    graphed = GraphedCallable(fn, DEVICE)
    graphed(("shift", 1))
    graphed(shift=1)
    graphed(1)
    graphed(True)
    graphed(1.0)
    graphed([1])
    graphed((1,))
    assert len(calls) == 7, "an equal-comparing call replayed another call's graph"


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


def test_an_eager_call_abandons_the_recording_and_the_callable_runs_plainly(caplog):
    """A recording that meets an eager call launches nothing; the call then runs plainly, now and on later calls."""
    before, between, after = _Affine(2, 0), _Affine(1, 10), _Affine(3, 0)

    def body(values: wp.array) -> wp.array:
        before(values)
        eager(between, values)
        return after(values)

    graphed = GraphedCallable(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    with caplog.at_level(logging.WARNING, logger="isaaclab.utils.warp.capture"):
        graphed(values)
    # only the plain run reached the device: 1 * 2 + 10 = 12 -> 36
    assert _read(values) == [36]
    graphed(values)
    # 36 * 2 + 10 = 82 -> 246; the abandoned recording ran the Python of `before` once more
    assert (_read(values), before.calls, between.calls, after.calls, _num_graphs(graphed)) == ([246], 3, 2, 2, 0)
    assert [record.getMessage() for record in caplog.records] == [
        "'test_an_eager_call_abandons_the_recording_and_the_callable_runs_plainly.<locals>.body' calls"
        " '_Affine', which cannot be recorded, so it runs without a CUDA graph."
    ]


def test_an_eager_call_abandons_the_recording_through_an_exception_handler():
    """The code a recording calls cannot catch the abandonment by mistake."""
    fn = _Affine(1, 10)

    def body(values: wp.array):
        try:
            eager(fn, values)
        except Exception:  # noqa: BLE001
            pytest.fail("the abandonment must pass an `except Exception` handler")

    graphed = GraphedCallable(body, DEVICE)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    graphed(values)
    assert (_read(values), fn.calls, _num_graphs(graphed)) == ([10], 1, 0)


def test_a_callable_that_ran_plainly_records_again_after_an_invalidation():
    """A term change can remove the eager call, so an invalidation lets the callable record again."""
    fn, eager_fn = _Affine(2, 0), _Affine(1, 10)
    uses_eager = [True]

    def body(values: wp.array):
        fn(values)
        if uses_eager[0]:
            eager(eager_fn, values)

    graphed = GraphedCallable(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)
    graphed(values)  # abandoned, then plain: 1 * 2 + 10 = 12
    uses_eager[0] = False
    GraphedCallable.invalidate()
    graphed(values)  # records: 24
    graphed(values)  # replays: 48
    assert (_read(values), fn.calls, _num_graphs(graphed)) == ([48], 3, 1)


def test_an_eager_call_of_a_nested_callable_makes_both_run_plainly():
    eager_fn, inner_fn, outer_fn = _Affine(1, 10), _Affine(2, 0), _Affine(3, 0)
    nested = GraphedCallable(lambda values: (eager(eager_fn, values), inner_fn(values)), DEVICE)
    outer = GraphedCallable(lambda values: (outer_fn(values), nested(values)), DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    outer(values)
    outer(values)
    # 1 * 3 + 10 = 13 -> 26, then 26 * 3 + 10 = 88 -> 176
    assert (_read(values), eager_fn.calls, inner_fn.calls) == ([176], 2, 2)
    assert (_num_graphs(outer), _num_graphs(nested), outer._plain, nested._plain) == (0, 0, True, True)


def test_eager_call_raises_inside_a_capture_no_callable_started():
    """A call that cannot be recorded must not be captured silently into a foreign graph."""
    fn = _Affine()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="no GraphedCallable started"):
        with wp.ScopedDevice(DEVICE), wp.ScopedCapture():
            eager(fn, values)
    assert fn.calls == 0


def test_capturing_sees_a_cpu_capture():
    """Warp's CPU capture counts as a capture, so a call that cannot be recorded raises inside it as on CUDA."""
    fn = _Affine()
    values = wp.zeros(1, dtype=wp.int32, device="cpu")

    with wp.ScopedDevice("cpu"), wp.ScopedCapture():
        assert capturing("cpu")
        with pytest.raises(RuntimeError, match="no GraphedCallable started"):
            eager(fn, values)
    assert not capturing("cpu")
    assert fn.calls == 0


def test_callable_raising_while_recording_ends_the_capture():
    """A failed recording launches nothing and must not leave the device capturing; the next call records again."""
    recorded = _Affine(2, 0)

    def body(values: wp.array) -> wp.array:
        recorded(values)
        if recorded.calls == 1:
            raise ValueError("first recording fails")
        return values

    graphed = GraphedCallable(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    with pytest.raises(ValueError, match="first recording fails"):
        graphed(values)
    assert not wp.get_device(DEVICE).is_capturing
    assert _read(values) == [1]
    graphed(values)
    assert (_read(values), _num_graphs(graphed)) == ([2], 1)


class _ChangeRegistry:
    """Host-side registry, like a solver's queue of model changes, that a recording cannot hold."""

    def __init__(self):
        self.changes = []

    @eager_method
    def add(self, change: int):
        self.changes.append(change)


def test_an_eager_method_makes_its_recorded_caller_run_plainly():
    """A writer whose host registration cannot be recorded runs plainly, so the registration happens on every call."""
    registry = _ChangeRegistry()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    fn = _Affine()

    def writer():
        fn(values)
        registry.add(7)

    graphed = GraphedCallable(writer, DEVICE)
    for _ in range(3):
        graphed()
    assert (registry.changes, _read(values), _num_graphs(graphed)) == ([7, 7, 7], [3], 0)


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


def _synchronizing_call(values: torch.Tensor) -> float:
    """A torch host synchronization, whose value a replay would keep."""
    return values.sum().item()


def test_a_host_synchronization_while_recording_raises_naming_the_call():
    """A torch operation that synchronizes the host inside a recorded graph raises instead of baking its value."""
    values = torch.ones(4, device=DEVICE)
    previous = torch.cuda.get_sync_debug_mode()
    graphed = GraphedCallable(lambda: _synchronizing_call(values), DEVICE)
    with pytest.raises(RuntimeError, match="<lambda>' synchronizes the host with the device or runs a torch CUDA"):
        graphed()
    assert torch.cuda.get_sync_debug_mode() == previous, "the sync debug mode must be restored after a raise"


def test_a_host_synchronization_in_an_eager_call_runs_plainly_in_the_previous_mode():
    values = torch.ones(4, device=DEVICE)
    previous = torch.cuda.get_sync_debug_mode()
    modes = []

    def body():
        modes.append(torch.cuda.get_sync_debug_mode())
        return eager(_synchronizing_call, values)

    graphed = GraphedCallable(body, DEVICE)
    assert graphed() == 4.0
    assert graphed() == 4.0
    # the abandoned recording ran in mode "error" (2), the plain runs in the previous mode
    assert modes == [2, previous, previous]
    assert torch.cuda.get_sync_debug_mode() == previous


def test_the_sync_debug_mode_is_restored_when_the_recording_raises():
    previous = torch.cuda.get_sync_debug_mode()

    def body():
        raise ValueError("term failed")

    with pytest.raises(ValueError, match="term failed"):
        GraphedCallable(body, DEVICE)()
    assert torch.cuda.get_sync_debug_mode() == previous


def test_a_torch_operation_while_recording_raises_naming_the_call_and_the_operation():
    """A torch operation on the recording's device would run once at recording time; the guard raises, with and
    without Kit, before CUDA would."""
    values = torch.ones(4, device=DEVICE)
    graphed = GraphedCallable(lambda: values.add_(1), DEVICE)
    with pytest.raises(RuntimeError, match=r"(?s)<lambda>' .*while it records.*aten\.add_"):
        graphed()
    # the operation did not run, and the guard is released after the raise
    assert values.add(1).sum().item() == 8.0


def test_a_torch_operation_in_an_eager_call_runs():
    values = torch.ones(4, device=DEVICE)
    graphed = GraphedCallable(lambda: eager(values.add_, 1), DEVICE)
    graphed()
    graphed()
    assert values.sum().item() == 12.0
