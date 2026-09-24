# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX render strategies: pipelining, delivery, staging slots, and teardown."""

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

    def step_async(self, render_products: set[str], delta_time: float) -> _PendingOp:
        op = _PendingOp(self._timeline, len(self.ops))
        self.ops.append(op)
        self._timeline.events.append(f"submit:{len(self.ops) - 1}")
        return op

    def step(self, render_products: set[str], delta_time: float) -> dict:
        self._timeline.events.append("step")
        return {}


# The renderer passes the same render data object for a camera on every frame. The tests do the
# same, since priming is tracked per camera.
_DEFAULT_CAMERA = object()


@pytest.fixture()
def timeline() -> _Timeline:
    return _Timeline()


@pytest.fixture()
def strategy() -> _AsyncRenderStrategy:
    strategy = _AsyncRenderStrategy()
    strategy.set_device("cuda:0")
    return strategy


def _render(
    strategy: Any, renderer: _FakeRenderer, ordinal: int, consumed: list[int], camera: object = _DEFAULT_CAMERA
) -> None:
    strategy.render(
        renderer,
        {"/Render/Product"},
        1.0 / 60.0,
        camera,
        lambda render_data, products: consumed.append(ordinal),
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
        "submit:0",
        "drain:0",
        "submit:1",
        "drain:1",
        "submit:2",
        "drain:2",
        "submit:3",
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

    strategy.render(renderer, {"/P"}, 1.0 / 60.0, first_target, consume)
    delivered.clear()

    strategy.render(renderer, {"/P"}, 1.0 / 60.0, second_target, consume)
    strategy.settle_before_scene_write()

    assert delivered == [second_target]


def test_released_render_data_is_not_delivered_into(strategy, timeline):
    """Per-camera cleanup disowns queued frames. Their ops still drain, but nothing delivers into
    the released buffers."""
    renderer = _FakeRenderer(timeline)
    camera = object()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.append(render_data)

    strategy.render(renderer, {"/P"}, 1.0 / 60.0, camera, consume)
    strategy.render(renderer, {"/P"}, 1.0 / 60.0, camera, consume)
    delivered.clear()

    strategy.release_render_data(camera)
    strategy.settle_before_scene_write()

    assert delivered == []
    assert all(op.waited for op in renderer.ops)


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
    camera_binding, object_binding = _FakeBinding(), _FakeBinding()
    consumed: list[int] = []

    camera_buffers = []
    object_buffers = []
    for ordinal in range(3):
        if camera_first:
            camera_buffers.append(_stage_camera(strategy, camera_binding))
            object_buffers.append(_stage_objects(strategy, object_binding))
        else:
            object_buffers.append(_stage_objects(strategy, object_binding))
            camera_buffers.append(_stage_camera(strategy, camera_binding))
        _render(strategy, renderer, ordinal, consumed)

    for buffers in (camera_buffers, object_buffers):
        assert buffers[0] is not buffers[1]
        assert buffers[0] is buffers[2]


@pytest.mark.parametrize("interleaved", [False, True], ids=["staged_then_rendered", "interleaved"])
def test_two_cameras_pipeline_together_with_one_frame_latency(timeline, interleaved):
    """Cameras share the strategy: both prime their first frame, and a frame's renders drain
    together when the next frame's renders are enqueued. Each camera stages into its own buffers.

    The interleaved order is the one the sensor pipeline produces: each camera stages its pose
    and renders before the next camera runs, and the object transforms stage once per step in
    between the first camera's pose and its render.
    """
    strategy = _AsyncRenderStrategy()
    strategy.set_device(wp.get_device("cuda:0"))
    renderer = _FakeRenderer(timeline)
    camera_a, camera_b = object(), object()
    binding_a, binding_b, binding_objects = _FakeBinding(), _FakeBinding(), _FakeBinding()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.append(render_data)

    def frame():
        buffer_a = _stage_camera(strategy, binding_a)
        if interleaved:
            _stage_objects(strategy, binding_objects)
            strategy.render(renderer, {"/A"}, 1.0 / 60.0, camera_a, consume)
            buffer_b = _stage_camera(strategy, binding_b)
        else:
            buffer_b = _stage_camera(strategy, binding_b)
            _stage_objects(strategy, binding_objects)
            strategy.render(renderer, {"/A"}, 1.0 / 60.0, camera_a, consume)
        strategy.render(renderer, {"/B"}, 1.0 / 60.0, camera_b, consume)
        return buffer_a, buffer_b

    frame_0 = frame()
    assert delivered == [camera_a, camera_b], "each camera's first frame is primed"
    assert frame_0[0] is not frame_0[1], "cameras must not share a staging buffer within a frame"

    delivered.clear()
    frame_1 = frame()
    assert delivered == [], "the second frame stays in flight"
    assert frame_1[0] is not frame_0[0], "consecutive frames must not share a staging buffer"

    delivered.clear()
    frame_2 = frame()
    assert delivered == [camera_a, camera_b], "a frame drains when the next frame is enqueued"
    assert frame_2[0] is frame_0[0], "staging buffers double-buffer across frames"


def test_repeated_renders_without_staging_keep_one_frame_in_flight(strategy, timeline):
    """A camera re-rendered without staging starts the next frame at the render call itself.

    Nothing is staged between the rounds, so the fallback boundary in ``_begin_render_phase``
    must group the renders into frames and drain the previous frame, for both cameras together.
    """
    renderer = _FakeRenderer(timeline)
    camera_a, camera_b = object(), object()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.append(render_data)

    def render_both():
        strategy.render(renderer, {"/A"}, 1.0 / 60.0, camera_a, consume)
        strategy.render(renderer, {"/B"}, 1.0 / 60.0, camera_b, consume)

    render_both()
    assert delivered == [camera_a, camera_b], "each camera's first frame is primed"

    delivered.clear()
    render_both()
    assert delivered == [], "the second round stays in flight"

    delivered.clear()
    strategy.render(renderer, {"/A"}, 1.0 / 60.0, camera_a, consume)
    assert delivered == [camera_a, camera_b], "a repeat render drains the whole previous round"
    strategy.render(renderer, {"/B"}, 1.0 / 60.0, camera_b, consume)
    assert delivered == [camera_a, camera_b], "the second camera joins the new round without draining it"


def test_cleanup_survives_failed_slot_writes(strategy, timeline):
    """A failed binding write at teardown must not raise. It must finish draining and report the failure."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []
    _render(strategy, renderer, 0, consumed)
    _render(strategy, renderer, 1, consumed)

    class _FailingWriteOp:
        def wait(self) -> None:
            raise RuntimeError("device lost")

    strategy._slots.append(_AsyncRenderSlot(write_ops=[_FailingWriteOp()]))
    errors = strategy.cleanup()

    assert consumed == [0, 1]
    assert len(errors) == 1


def test_sync_strategy_needs_no_barrier(timeline):
    """The barrier is a no-op for synchronous rendering, which holds nothing in flight."""
    strategy = _SyncRenderStrategy()
    strategy.set_device("cuda:0")
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    strategy.settle_before_scene_write()
    _render(strategy, renderer, 7, consumed)
    strategy.settle_before_scene_write()

    assert timeline.events == ["step"]
    assert consumed == [7]
