# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the scene-write barrier that bounds pipelined renders under ovstage."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_renderer_strategies import (
        _AsyncRenderSlot,
        _AsyncRenderStrategy,
        _SyncRenderStrategy,
    )


class _Timeline:
    """Records the order of renderer and stage events so their interleaving can be asserted."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def of_kind(self, kind: str) -> list[str]:
        return [event for event in self.events if event.startswith(kind)]


class _PendingOp:
    """An ovrtx step operation that only reports completion once waited on."""

    def __init__(self, timeline: _Timeline, index: int) -> None:
        self._timeline = timeline
        self._index = index
        self.waited = False

    def wait(self) -> _PendingOp:
        self.waited = True
        self._timeline.events.append(f"drain:{self._index}")
        return self

    def fetch(self) -> dict:
        return {}


class _FakeRenderer:
    """Renderer stub that records each submitted step and hands back a pending operation."""

    def __init__(self, timeline: _Timeline) -> None:
        self._timeline = timeline
        self.ops: list[_PendingOp] = []

    def step_async(self, render_products: set[str], delta_time: float, *, ordinal: int | None = None) -> _PendingOp:
        op = _PendingOp(self._timeline, len(self.ops))
        self.ops.append(op)
        self._timeline.events.append(f"submit:{len(self.ops) - 1}:ordinal={ordinal}")
        return op

    def step(self, render_products: set[str], delta_time: float, *, ordinal: int | None = None) -> dict:
        self._timeline.events.append(f"step:ordinal={ordinal}")
        return {}


@pytest.fixture()
def timeline() -> _Timeline:
    return _Timeline()


@pytest.fixture()
def strategy() -> _AsyncRenderStrategy:
    strategy = _AsyncRenderStrategy()
    strategy.set_device("cuda:0")
    return strategy


def _render(strategy: Any, renderer: _FakeRenderer, ordinal: int, consumed: list[int]) -> None:
    strategy.render(
        renderer,
        {"/Render/Product"},
        1.0 / 60.0,
        object(),
        lambda render_data, products: consumed.append(ordinal),
        ordinal=ordinal,
    )


def test_settle_drains_every_in_flight_render(strategy, timeline):
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    _render(strategy, renderer, 0, consumed)
    _render(strategy, renderer, 1, consumed)
    strategy.settle_before_scene_write()

    assert all(op.waited for op in renderer.ops)
    assert not strategy._has_pending_ops()


def test_settle_is_idempotent_without_pending_renders(strategy, timeline):
    renderer = _FakeRenderer(timeline)

    strategy.settle_before_scene_write()
    strategy.settle_before_scene_write()

    assert timeline.events == []
    assert renderer.ops == []


def test_render_stays_pipelined_when_settle_precedes_each_frame(strategy, timeline):
    """A submit must precede the drain of that same frame, otherwise the path is merely synchronous."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    for ordinal in range(4):
        strategy.settle_before_scene_write()
        _render(strategy, renderer, ordinal, consumed)

    # Frame 0 is primed synchronously; frames 1..3 each drain only at the following frame's write.
    assert timeline.events == [
        "submit:0:ordinal=0",
        "drain:0",
        "submit:1:ordinal=1",
        "drain:1",
        "submit:2:ordinal=2",
        "drain:2",
        "submit:3:ordinal=3",
    ]
    assert consumed == [0, 1, 2]


def test_every_frame_is_delivered_exactly_once(strategy, timeline):
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    for ordinal in range(5):
        strategy.settle_before_scene_write()
        _render(strategy, renderer, ordinal, consumed)
    strategy.cleanup()

    assert consumed == [0, 1, 2, 3, 4]


def test_first_frame_is_primed_after_reinitialize(strategy, timeline):
    """Priming must be tracked explicitly: a drained ring still means 'already primed'."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    _render(strategy, renderer, 0, consumed)
    assert consumed == [0]

    strategy.initialize(4)
    timeline.events.clear()
    consumed.clear()

    _render(strategy, renderer, 1, consumed)
    assert consumed == [1], "the first frame of a new scene must be primed synchronously"


def test_frames_are_delivered_to_the_render_data_they_were_submitted_for(strategy, timeline):
    """A drain triggered by a scene write must not deliver into another frame's buffers."""
    renderer = _FakeRenderer(timeline)
    first_target = object()
    second_target = object()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.append(render_data)

    strategy.render(renderer, {"/P"}, 1.0 / 60.0, first_target, consume, ordinal=0)
    delivered.clear()

    strategy.render(renderer, {"/P"}, 1.0 / 60.0, second_target, consume, ordinal=1)
    strategy.settle_before_scene_write()

    assert delivered == [second_target]


class _CompletedWriteOp:
    """A binding write op that is already complete."""

    def wait(self) -> None:
        return None


class _FakeBinding:
    """Accepts async binding writes and completes them immediately."""

    def write_async(self, data, **_kwargs) -> _CompletedWriteOp:
        return _CompletedWriteOp()


def _stage_camera(strategy: _AsyncRenderStrategy, binding: _FakeBinding) -> Any:
    with strategy.stage_camera_transforms(binding, 2) as (_quats, transforms):
        return transforms


def _stage_objects(strategy: _AsyncRenderStrategy, binding: _FakeBinding) -> Any:
    with strategy.stage_object_transforms(binding, 2, None) as transforms:
        return transforms


@pytest.mark.parametrize("camera_first", [True, False], ids=["camera_first", "objects_first"])
def test_staged_buffers_are_double_buffered_per_frame(timeline, camera_first):
    """Camera and object updates share one slot per frame, in either order: the buffers staged in
    frame N are reused in frame N+2, never in frame N+1, whose render is still in flight."""
    strategy = _AsyncRenderStrategy()
    strategy.set_device(wp.get_device("cuda:0"))
    strategy.initialize(2)
    renderer = _FakeRenderer(timeline)
    binding = _FakeBinding()
    consumed: list[int] = []

    camera_buffers = []
    object_buffers = []
    for ordinal in range(3):
        if camera_first:
            camera_buffers.append(_stage_camera(strategy, binding))
            object_buffers.append(_stage_objects(strategy, binding))
        else:
            object_buffers.append(_stage_objects(strategy, binding))
            camera_buffers.append(_stage_camera(strategy, binding))
        _render(strategy, renderer, ordinal, consumed)

    for buffers in (camera_buffers, object_buffers):
        assert buffers[0] is not buffers[1]
        assert buffers[0] is buffers[2]


def test_cleanup_survives_failed_slot_writes(strategy, timeline):
    """A failed binding write at teardown must log and continue, not raise out of the renderer's close()."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []
    _render(strategy, renderer, 0, consumed)
    _render(strategy, renderer, 1, consumed)

    class _FailingWriteOp:
        def wait(self) -> None:
            raise RuntimeError("device lost")

    strategy._slots.append(
        _AsyncRenderSlot(
            camera_transforms=None, camera_quats=None, object_transforms=None, write_ops=[_FailingWriteOp()]
        )
    )
    strategy.cleanup()

    assert consumed == [0, 1]


def test_sync_strategy_needs_no_barrier(timeline):
    """The barrier is a no-op for synchronous rendering, which holds nothing in flight."""
    strategy = _SyncRenderStrategy()
    strategy.set_device("cuda:0")
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    strategy.settle_before_scene_write()
    _render(strategy, renderer, 7, consumed)
    strategy.settle_before_scene_write()

    assert timeline.events == ["step:ordinal=7"]
    assert consumed == [7]
