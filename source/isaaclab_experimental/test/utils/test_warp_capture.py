# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the capture of Warp frontend stages."""

from __future__ import annotations

import pytest
import torch
import warp as wp
from isaaclab_experimental.utils.warp_capture import (
    SYNC_DEBUG_ENV_VAR,
    CapturedStage,
    captured,
    eager,
    reset_captured_stages,
)

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"


@wp.kernel
def _affine(values: wp.array(dtype=wp.int32), scale: wp.int32, shift: wp.int32):
    values[wp.tid()] = values[wp.tid()] * scale + shift


class _AffineStage:
    """Stage that applies ``x * scale + shift`` on the device and counts its Python invocations."""

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
    monkeypatch.setattr(CapturedStage, "enabled", True)


def test_first_call_records_and_runs_the_stage_once():
    """Recording has no eager warm-up: a stateful stage advances once per call, including the first."""
    fn = _AffineStage()
    stage = CapturedStage(fn, DEVICE)
    values = wp.zeros(2, dtype=wp.int32, device=DEVICE)

    assert stage(values) is values
    assert _read(values) == [1, 1]
    assert stage(values) is values
    assert _read(values) == [2, 2]
    assert (fn.calls, stage.num_graphs) == (1, 1), "the second call must replay the recorded graph"


def test_changed_argument_records_the_stage_again():
    """A graph replays its recorded pointers and scalars, so a new array or scalar records a new graph."""
    fn = _AffineStage()
    stage = CapturedStage(fn, DEVICE)
    old, new = wp.zeros(1, dtype=wp.int32, device=DEVICE), wp.zeros(1, dtype=wp.int32, device=DEVICE)

    stage(old)
    stage(new)
    stage(new)
    stage(new, 5)
    assert (_read(old), _read(new), fn.calls, stage.num_graphs) == ([1], [3], 3, 3)


def test_physics_rebind_records_the_stage_again():
    fn = _AffineStage()
    stage = CapturedStage(fn, DEVICE)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    stage(values)

    CapturedStage.invalidate()
    stage(values)
    stage(values)
    assert (fn.calls, _read(values), stage.num_graphs) == (2, [3], 1)


def test_stage_runs_eagerly_when_disabled_or_on_cpu(monkeypatch):
    on_cpu = _AffineStage()
    cpu_stage = CapturedStage(on_cpu, "cpu")
    cpu_values = wp.zeros(1, dtype=wp.int32, device="cpu")
    cpu_stage(cpu_values)
    cpu_stage(cpu_values)

    monkeypatch.setattr(CapturedStage, "enabled", False)
    disabled = _AffineStage()
    stage = CapturedStage(disabled, DEVICE)
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    stage(values)
    stage(values)

    assert (on_cpu.calls, _read(cpu_values), cpu_stage.num_graphs) == (2, [2], 0)
    assert (disabled.calls, _read(values), stage.num_graphs) == (2, [2], 0)


def test_eager_call_runs_between_its_graphs_on_every_call():
    """The recording runs the graph before an eager call first, and every replay keeps that order."""
    before, between, after = _AffineStage(2, 0), _AffineStage(1, 10), _AffineStage(3, 0)

    def body(values: wp.array) -> wp.array:
        before(values)
        eager(between, values)
        return after(values)

    stage = CapturedStage(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    stage(values)
    # the non-commuting steps only give these values in order: 1 * 2 + 10 = 12 -> 36, then 36 * 2 + 10 = 82 -> 246
    assert _read(values) == [36]
    stage(values)
    assert (_read(values), between.calls, before.calls, after.calls, stage.num_graphs) == ([246], 2, 1, 1, 2)


def test_eager_call_first_leaves_one_graph_after_it():
    """A stage making its eager call first records all its capturable work into the single graph after it."""
    eager_fn, first, second = _AffineStage(1, 10), _AffineStage(2, 0), _AffineStage(3, 0)

    def body(values: wp.array) -> wp.array:
        eager(eager_fn, values)
        first(values)
        return second(values)

    stage = CapturedStage(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)
    stage(values)
    stage(values)

    assert _read(values) == [((1 + 10) * 6 + 10) * 6]
    ((steps, _),) = stage._recordings.values()
    assert [isinstance(step, wp.Graph) for step in steps] == [True, False, True]
    # the graph before the eager call is empty; the last one multiplies by 6
    wp.capture_launch(steps[0])
    assert _read(values) == [456]
    wp.capture_launch(steps[-1])
    assert _read(values) == [456 * 6]


class _Owner:
    device = DEVICE

    def __init__(self):
        self.inner_fn, self.outer_fn = _AffineStage(2, 0), _AffineStage(1, 1)

    @captured
    def inner(self, values: wp.array) -> wp.array:
        return self.inner_fn(values)

    @captured
    def outer(self, values: wp.array) -> wp.array:
        self.outer_fn(values)
        return self.inner(values)


def test_decorated_methods_own_their_stages_and_nest_into_an_enclosing_recording():
    owner = _Owner()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    assert owner.outer(values) is values
    owner.outer(values)
    # (0 + 1) * 2 = 2 -> (2 + 1) * 2 = 6: the inner call is part of the outer graph
    assert _read(values) == [6]
    assert (owner.outer_fn.calls, owner.inner_fn.calls) == (1, 1)
    stages = owner._captured_stages
    assert (stages[_Owner.outer].num_graphs, stages[_Owner.inner].num_graphs) == (1, 0)

    reset_captured_stages(owner)
    owner.outer(values)
    assert (owner.outer_fn.calls, _read(values)) == (2, [14])


def test_eager_call_of_a_nested_stage_splits_the_enclosing_recording():
    eager_fn, inner_fn, outer_fn = _AffineStage(1, 10), _AffineStage(2, 0), _AffineStage(3, 0)
    nested = CapturedStage(lambda values: (eager(eager_fn, values), inner_fn(values)), DEVICE)
    outer = CapturedStage(lambda values: (outer_fn(values), nested(values)), DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    outer(values)
    outer(values)
    # 1 * 3 + 10 = 13 -> 26, then 26 * 3 + 10 = 88 -> 176: the nested eager call runs between the outer graphs
    assert (_read(values), eager_fn.calls, inner_fn.calls, outer_fn.calls) == ([176], 2, 1, 1)
    assert (outer.num_graphs, nested.num_graphs) == (2, 0)


def test_eager_call_raises_inside_a_capture_no_stage_started():
    """A call that cannot be recorded must not be captured silently into a foreign graph."""
    fn = _AffineStage()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="no CapturedStage started"):
        with wp.ScopedDevice(DEVICE), wp.ScopedCapture():
            eager(fn, values)
    assert fn.calls == 0


def test_stage_raising_while_recording_ends_the_capture():
    """A failed recording must not leave the device capturing; the next call records again."""
    recorded, eager_fn = _AffineStage(2, 0), _AffineStage(1, 10)

    def body(values: wp.array) -> wp.array:
        recorded(values)
        eager(eager_fn, values)
        if recorded.calls == 1:
            raise ValueError("first recording fails")
        return values

    stage = CapturedStage(body, DEVICE)
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)

    with pytest.raises(ValueError, match="first recording fails"):
        stage(values)
    assert not wp.get_device(DEVICE).is_capturing
    stage(values)
    # the failed recording ran its first graph and the eager call: 1 * 2 + 10 = 12, then 12 * 2 + 10 = 34
    assert (_read(values), stage.num_graphs) == ([34], 2)


def test_sync_debug_trap_raises_on_a_hidden_host_sync(monkeypatch):
    monkeypatch.setattr(CapturedStage, "enabled", False)
    monkeypatch.setenv(SYNC_DEBUG_ENV_VAR, "1")
    flags = torch.ones(4, dtype=torch.bool, device=DEVICE)
    stage = CapturedStage(lambda: flags.any().item(), DEVICE)

    with pytest.raises(RuntimeError, match="synchroniz"):
        stage()
    assert torch.cuda.get_sync_debug_mode() == 0, "the trap must be released after the stage"
