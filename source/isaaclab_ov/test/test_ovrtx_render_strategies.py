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


class _CompletedWriteOp:
    """A binding write op that is already complete."""

    def wait(self) -> None:
        return None


class _FakeBinding:
    """Accepts async binding writes and completes them immediately."""

    def write_async(self, data, **_kwargs) -> _CompletedWriteOp:
        return _CompletedWriteOp()


@pytest.fixture()
def timeline() -> _Timeline:
    return _Timeline()


@pytest.fixture()
def strategy() -> _AsyncRenderStrategy:
    strategy = _AsyncRenderStrategy()
    strategy.set_device(wp.get_device("cuda:0"))
    return strategy


# The renderer passes the same render data object for a camera on every frame. The tests do the
# same, since priming is tracked per camera.
_DEFAULT_CAMERA = object()


def _render(
    strategy: Any, renderer: _FakeRenderer, ordinal: int, consumed: list[int], camera: object = _DEFAULT_CAMERA
) -> None:
    strategy.render(
        renderer,
        {"/Render/Product"},
        1.0 / 60.0,
        (camera,),
        lambda render_data, products: consumed.append(ordinal),
    )


def _stage_camera(strategy: _AsyncRenderStrategy, binding: _FakeBinding) -> Any:
    with strategy.stage_camera_transforms(binding, 2) as (_quats, transforms):
        return transforms


def _stage_objects(strategy: _AsyncRenderStrategy, binding: _FakeBinding) -> Any:
    with strategy.stage_object_transforms(binding, 2, None) as transforms:
        return transforms


def test_unannounced_renders_are_synchronous(strategy, timeline):
    """Without announce_frame there is no boundary to deliver at, so every render delivers itself."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    _render(strategy, renderer, 0, consumed)
    _render(strategy, renderer, 1, consumed)

    assert consumed == [0, 1]
    assert not strategy._has_pending_ops()


def test_settle_drains_every_in_flight_render(strategy, timeline):
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    strategy.announce_frame(0)
    _render(strategy, renderer, 0, consumed)
    strategy.announce_frame(1)
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
        strategy.announce_frame(ordinal)
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
        strategy.announce_frame(ordinal)
        _render(strategy, renderer, ordinal, consumed)
    strategy.cleanup()

    assert consumed == [0, 1, 2, 3, 4]


def test_first_frame_is_primed_after_reinitialize(strategy, timeline):
    """Priming must be tracked explicitly: a drained ring still means 'already primed'."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    strategy.announce_frame(0)
    _render(strategy, renderer, 0, consumed)
    assert consumed == [0]

    strategy.initialize(4)
    timeline.events.clear()
    consumed.clear()

    strategy.announce_frame(1)
    _render(strategy, renderer, 1, consumed)
    assert consumed == [1], "the first frame of a new scene must be primed synchronously"


def test_frames_are_delivered_to_the_render_data_they_were_submitted_for(strategy, timeline):
    """Priming delivers into the batch it was submitted for, never into another camera's buffers."""
    renderer = _FakeRenderer(timeline)
    first_target = object()
    second_target = object()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.append(render_data)

    strategy.announce_frame(0)
    strategy.render(renderer, {"/P"}, 1.0 / 60.0, (first_target,), consume)
    delivered.clear()

    strategy.render(renderer, {"/P"}, 1.0 / 60.0, (second_target,), consume)
    strategy.settle_before_scene_write()

    assert delivered == [(second_target,)]


def test_lazy_per_camera_renders_share_the_announced_frame(strategy, timeline):
    """Cameras rendered one call at a time still pipeline fully when each step announces its
    frame: all of a frame's renders stay in flight and drain together at the next frame."""
    renderer = _FakeRenderer(timeline)
    camera_a, camera_b = object(), object()
    binding_a, binding_b = _FakeBinding(), _FakeBinding()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.extend(render_data)

    def step(index: int) -> None:
        strategy.announce_frame(index)
        _stage_camera(strategy, binding_a)
        strategy.render(renderer, {"/A"}, 1.0 / 60.0, (camera_a,), consume)
        _stage_camera(strategy, binding_b)
        strategy.render(renderer, {"/B"}, 1.0 / 60.0, (camera_b,), consume)

    step(0)
    assert delivered == [camera_a, camera_b], "each camera's first frame is primed"

    delivered.clear()
    step(1)
    assert delivered == [], "the whole frame stays in flight"

    step(2)
    assert delivered == [camera_a, camera_b], "the frame's renders drain together at the next frame"


def test_one_batch_per_frame_pipelines_all_cameras(strategy, timeline):
    """The eager render context submits all cameras of a step as one batch."""
    renderer = _FakeRenderer(timeline)
    camera_a, camera_b = object(), object()
    binding_a, binding_b = _FakeBinding(), _FakeBinding()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.extend(render_data)

    def step(index: int) -> None:
        strategy.announce_frame(index)
        _stage_camera(strategy, binding_a)
        _stage_camera(strategy, binding_b)
        strategy.render(renderer, {"/A", "/B"}, 1.0 / 60.0, (camera_a, camera_b), consume)

    step(0)
    assert delivered == [camera_a, camera_b], "the first batch is primed"
    delivered.clear()
    step(1)
    assert delivered == [], "the second batch stays in flight"
    step(2)
    assert delivered == [camera_a, camera_b], "a batch drains when the next frame's batch is enqueued"


def test_repeat_render_within_a_frame_delivers_the_previous_entry(strategy, timeline):
    """A camera rendered again in one frame delivers its previous entry, so a caller that renders
    without advancing the frame cannot grow the ring."""
    renderer = _FakeRenderer(timeline)
    camera = object()
    consumed: list[int] = []

    strategy.announce_frame(0)
    _render(strategy, renderer, 0, consumed, camera)  # primed and delivered
    _render(strategy, renderer, 1, consumed, camera)  # queued
    assert consumed == [0]

    _render(strategy, renderer, 2, consumed, camera)
    assert consumed == [0, 1], "the repeat delivers the camera's previous entry"
    assert strategy._has_pending_ops()


def test_frame_change_without_staging_rotates_at_most_once(strategy, timeline):
    """Announced frames with no staging keep the slot: rotating on every index change would land
    staging back on the slot that backs the renders in flight."""
    renderer = _FakeRenderer(timeline)
    camera = object()
    binding = _FakeBinding()
    consumed: list[int] = []

    strategy.announce_frame(0)
    frame_0 = _stage_camera(strategy, binding)
    _render(strategy, renderer, 0, consumed, camera)

    strategy.announce_frame(1)
    strategy.announce_frame(2)
    frame_2 = _stage_camera(strategy, binding)
    assert frame_2 is not frame_0, "the first staging after a staged frame uses the other slot"

    strategy.announce_frame(3)
    frame_3 = _stage_camera(strategy, binding)
    assert frame_3 is frame_0, "the slots keep alternating once per staged frame"


def test_double_staging_one_frame_reuses_the_slot_after_waiting_writes(strategy, timeline):
    """Re-staging a binding within one frame reuses its buffer after the slot's pending write ops
    are waited out, so the pending ingest cannot read a half-refilled buffer."""

    class _RecordingWriteOp:
        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            self.waited = True

    class _RecordingBinding:
        def __init__(self) -> None:
            self.ops: list[_RecordingWriteOp] = []

        def write_async(self, data, **_kwargs) -> _RecordingWriteOp:
            op = _RecordingWriteOp()
            self.ops.append(op)
            return op

    binding = _RecordingBinding()
    strategy = _AsyncRenderStrategy()
    strategy.set_device(wp.get_device("cuda:0"))

    strategy.announce_frame(0)
    first = _stage_camera(strategy, binding)
    second = _stage_camera(strategy, binding)

    assert second is first, "a second staging of one frame reuses its buffer"
    assert binding.ops[0].waited, "the pending write op is waited out before the refill"


def test_staged_buffers_are_double_buffered_per_frame(timeline):
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
        strategy.announce_frame(ordinal)
        camera_buffers.append(_stage_camera(strategy, camera_binding))
        object_buffers.append(_stage_objects(strategy, object_binding))
        _render(strategy, renderer, ordinal, consumed)

    for buffers in (camera_buffers, object_buffers):
        assert buffers[0] is not buffers[1]
        assert buffers[0] is buffers[2]


def test_release_camera_delivers_pending_frames_and_reprimes(strategy, timeline):
    """Releasing a camera delivers its queued frame while the buffers are still valid, and a
    camera re-created with the same buffers primes again."""
    renderer = _FakeRenderer(timeline)
    camera = object()
    binding = _FakeBinding()
    delivered: list[object] = []

    def consume(render_data, products):
        delivered.extend(render_data)

    strategy.announce_frame(0)
    strategy.render(renderer, {"/P"}, 1.0 / 60.0, (camera,), consume)  # primed and delivered
    strategy.announce_frame(1)
    strategy.render(renderer, {"/P"}, 1.0 / 60.0, (camera,), consume)  # queued
    delivered.clear()

    strategy.release_camera(camera, binding)

    assert delivered == [camera], "the queued frame is delivered before the release"
    assert all(op.waited for op in renderer.ops)
    assert not strategy._has_pending_ops()

    delivered.clear()
    strategy.announce_frame(2)
    strategy.render(renderer, {"/P"}, 1.0 / 60.0, (camera,), consume)
    assert delivered == [camera], "a re-registered camera primes again"


def test_released_binding_drops_its_staging_buffers(timeline):
    """Releasing a camera's binding evicts its cached buffers, so a recycled ``id()`` cannot
    reuse them."""
    strategy = _AsyncRenderStrategy()
    strategy.set_device(wp.get_device("cuda:0"))
    binding = _FakeBinding()

    strategy.announce_frame(0)
    first = _stage_camera(strategy, binding)
    strategy.release_camera(object(), binding)
    strategy.announce_frame(1)
    second = _stage_camera(strategy, binding)
    assert second is not first, "released bindings must not keep staging buffers alive"


def test_cleanup_survives_failed_slot_writes(strategy, timeline):
    """A failed binding write at teardown must not raise. It must finish draining and report the failure."""
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []
    strategy.announce_frame(0)
    _render(strategy, renderer, 0, consumed)
    strategy.announce_frame(1)
    _render(strategy, renderer, 1, consumed)

    class _FailingWriteOp:
        def wait(self) -> None:
            raise RuntimeError("device lost")

    strategy._slots.append(_AsyncRenderSlot(write_ops=[_FailingWriteOp()]))
    errors = strategy.cleanup()

    assert consumed == [0, 1]
    assert len(errors) == 1


@pytest.mark.parametrize("next_frame", [1, 2], ids=["same-frame", "next-frame"])
def test_cleanup_retains_new_render_after_previous_delivery_fails(strategy, timeline, next_frame):
    """Output extraction failure must not orphan the render submitted just before it."""
    renderer = _FakeRenderer(timeline)
    consumed = []
    strategy.announce_frame(0)
    _render(strategy, renderer, 0, consumed)

    def fail_delivery(render_data, products):
        raise RuntimeError("output extraction failed")

    strategy.announce_frame(1)
    strategy.render(renderer, {"/Render/Product"}, 1 / 60, (_DEFAULT_CAMERA,), fail_delivery)
    strategy.announce_frame(next_frame)
    with pytest.raises(RuntimeError, match="output extraction failed"):
        _render(strategy, renderer, 2, consumed)

    assert strategy.cleanup() == []
    assert all(op.waited for op in renderer.ops)
    assert consumed == [0, 2]


def test_sync_strategy_needs_no_barrier(timeline):
    """The barrier is a no-op for synchronous rendering, which holds nothing in flight."""
    strategy = _SyncRenderStrategy()
    strategy.set_device("cuda:0")
    renderer = _FakeRenderer(timeline)
    consumed: list[int] = []

    strategy.settle_before_scene_write()
    strategy.announce_frame(0)
    _render(strategy, renderer, 7, consumed)
    strategy.settle_before_scene_write()

    assert timeline.events == ["step"]
    assert consumed == [7]
