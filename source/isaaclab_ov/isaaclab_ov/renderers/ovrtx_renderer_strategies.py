# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering execution strategies for the OVRTX renderer.

The renderer delegates *how* a frame is executed -- how transform writes are staged and how the OVRTX
step is dispatched and consumed -- to a :class:`_RenderStrategy`, selected once from configuration so
the renderer's call sites carry no ``sync``/``async`` branching.

- :class:`_SyncRenderStrategy` writes transforms straight into OVRTX from caller-owned buffers and
  consumes each step inline.
- :class:`_AsyncRenderStrategy` pipelines steps and double-buffers transform staging, so rendering
  overlaps simulation at the cost of camera outputs arriving one frame later.

The interface is mechanism-neutral (:meth:`~_RenderStrategy.initialize`,
:meth:`~_RenderStrategy.stage_object_transforms`, :meth:`~_RenderStrategy.stage_camera_transforms`,
:meth:`~_RenderStrategy.render`, :meth:`~_RenderStrategy.cleanup`); slot and queue vocabulary stays
private to :class:`_AsyncRenderStrategy`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import warp as wp
from ovrtx import DataAccess

from isaaclab.renderers import resolve_async_rendering_enabled

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ovrtx import Operation, PendingFetch, Renderer, RenderProductSetOutputs

    from .ovrtx_renderer import OVRTXRenderData
    from .ovrtx_renderer_cfg import OVRTXRendererCfg

    _AsyncRenderOp: TypeAlias = Operation[PendingFetch[RenderProductSetOutputs]]
    _RenderProductConsumer: TypeAlias = Callable[[OVRTXRenderData, RenderProductSetOutputs], None]


# OVRTX steps assume a fixed frame delta for temporal accumulation and motion vectors.
_RENDER_DELTA_TIME = 1.0 / 60.0


class _AsyncRenderEntry:
    """One queued render step and the camera data that receives its output.

    The entry stores its own destination. Any caller can therefore drain the queue, and the frame
    still arrives in the buffers it was rendered for.
    """

    def __init__(
        self,
        op: _AsyncRenderOp,
        render_data: OVRTXRenderData | None,
        consume_products: _RenderProductConsumer,
    ) -> None:
        self.op = op
        self.render_data = render_data
        self.consume_products = consume_products

    def deliver(self) -> bool:
        """Wait for the render, then deliver its products to the stored destination."""
        products = self.op.wait().fetch()
        if products is None:
            return False
        if self.render_data is not None:
            self.consume_products(self.render_data, products)
        return True


class _RenderStrategy(ABC):
    """Decides how the OVRTX renderer stages transforms and runs render steps.

    :class:`_SyncRenderStrategy` renders and reads back within one call.
    :class:`_AsyncRenderStrategy` queues renders and reads them back one frame later. The renderer
    drives either one through this interface and never branches on the mode.
    """

    def __init__(self) -> None:
        self._warp_device: wp.Device | None = None

    def set_device(self, warp_device: wp.Device) -> None:
        """Record the renderer's resolved Warp device used for staging buffers and sync streams."""
        self._warp_device = warp_device

    @property
    def _cuda_stream(self) -> int:
        """The device's current Warp CUDA stream. OVRTX orders its reads against this stream.

        Always read this at use time. The current stream can change, for example during CUDA graph
        capture. The device cannot.
        """
        return self._warp_device.stream.cuda_stream

    def initialize(self, num_envs: int) -> None:
        """Prepare per-scene staging resources for ``num_envs`` environments.

        The renderer calls this once per scene initialization. The default does nothing.
        """

    def cleanup(self) -> list[Exception]:
        """Finish all queued work, release staging resources, and return the collected failures.

        This method must not raise. The caller re-raises the failures after its own teardown is
        done. The default does nothing.
        """
        return []

    def release_render_data(self, render_data: OVRTXRenderData) -> None:
        """Stop delivering queued frames into ``render_data``.

        A camera calls this when it releases its buffers. Queued renders still complete, but they
        no longer deliver into the released buffers. The default does nothing.
        """

    def settle_before_scene_write(self) -> None:
        """Wait until no render is in flight. Call this before writing to the scene.

        A queued render may read scene storage while it runs. A scene write during that read can
        corrupt the frame. The default does nothing.
        """

    @abstractmethod
    def stage_object_transforms(
        self, binding: Any, num_rows: int, buffer: wp.array
    ) -> AbstractContextManager[wp.array]:
        """Provide a ``mat44d`` buffer for ``num_rows`` object transforms.

        Use as a context manager and fill the yielded buffer inside the block. On a clean exit, the
        strategy writes the buffer to ``binding``. ``buffer`` is the caller's persistent staging
        array. A strategy may yield its own buffer instead.

        Enqueue the fill kernels on the device's current Warp stream, the default for ``wp.launch``
        and ``wp.copy``. The write orders OVRTX's read against that stream.
        """

    @abstractmethod
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> AbstractContextManager[tuple[wp.array, wp.array]]:
        """Provide ``(quats, transforms)`` staging buffers for ``num_rows`` cameras.

        Use as a context manager. ``quats`` is ``quatf`` scratch space. On a clean exit, the
        strategy writes the ``mat44d`` ``transforms`` buffer to ``binding``. The Warp-stream rule
        of :meth:`stage_object_transforms` applies here too.
        """

    @abstractmethod
    def render(
        self,
        renderer: Renderer,
        render_products: set[str],
        delta_time: float,
        render_data: OVRTXRenderData,
        consume_products: _RenderProductConsumer,
        ordinal: int | None = None,
    ) -> None:
        """Render one frame and deliver its products, either now or later.

        ``ordinal`` names the ovstage publication the render must observe. Pass it while an ovstage
        is attached to ``renderer``. Pass ``None`` otherwise.
        """


class _SyncRenderStrategy(_RenderStrategy):
    """Renders within the call: transforms go straight to OVRTX, and each step is read back inline.

    Transform writes use the blocking ``write()``, so a staged buffer stays valid until OVRTX has read
    it. ``DataAccess.ASYNC`` with the producing Warp stream lets OVRTX read GPU buffers in place.
    OVRTX rejects ``SYNC`` for GPU buffers.
    """

    @contextmanager
    def stage_object_transforms(self, binding: Any, num_rows: int, buffer: wp.array) -> Iterator[wp.array]:
        """Yield the caller's persistent buffer and write it to ``binding`` on exit.

        See :meth:`_RenderStrategy.stage_object_transforms`.
        """
        yield buffer
        binding.write(buffer, data_access=DataAccess.ASYNC, cuda_stream=self._cuda_stream)

    @contextmanager
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> Iterator[tuple[wp.array, wp.array]]:
        """Yield freshly allocated staging buffers and write the transforms to ``binding`` on exit.

        See :meth:`_RenderStrategy.stage_camera_transforms`.
        """
        camera_quats = wp.empty(num_rows, dtype=wp.quatf, device=self._warp_device)
        camera_transforms = wp.zeros(num_rows, dtype=wp.mat44d, device=self._warp_device)
        yield camera_quats, camera_transforms
        binding.write(camera_transforms, data_access=DataAccess.ASYNC, cuda_stream=self._cuda_stream)

    def render(
        self,
        renderer: Renderer,
        render_products: set[str],
        delta_time: float,
        render_data: OVRTXRenderData,
        consume_products: _RenderProductConsumer,
        ordinal: int | None = None,
    ) -> None:
        """Step OVRTX and consume its products inline. See :meth:`_RenderStrategy.render`."""
        products = renderer.step(render_products=render_products, delta_time=delta_time, ordinal=ordinal)
        consume_products(render_data, products)


@dataclass
class _AsyncRenderSlot:
    """A reusable set of transform staging buffers for one in-flight async update."""

    camera_transforms: wp.array
    camera_quats: wp.array
    object_transforms: wp.array | None
    write_ops: list[Operation]

    def record_write(self, binding: Any, data: wp.array, cuda_stream: int) -> None:
        """Write ``data`` to ``binding`` asynchronously and remember the write op.

        ``cuda_stream`` is the Warp stream that filled ``data``. OVRTX waits on that stream before
        it reads ``data``, so the read never sees a half-written buffer. The OVRTX API requires
        this stream handoff for GPU data.
        """
        self.write_ops.append(binding.write_async(data, data_access=DataAccess.ASYNC, cuda_stream=cuda_stream))

    def wait_for_writes(self) -> None:
        """Wait until this slot's async writes are done, so its buffers are safe to reuse."""
        if not self.write_ops:
            return
        try:
            for op in self.write_ops:
                op.wait()
        except Exception as e:
            raise RuntimeError("Failed to complete OVRTX async binding write before slot reuse") from e
        finally:
            self.write_ops = []


class _AsyncRenderStrategy(_RenderStrategy):
    """Queues render steps and reads them back one frame later, with double-buffered staging.

    Rendering then overlaps the next step's simulation and Python work. Camera outputs are one
    step stale.

    Transform staging always uses two slots. A finished write op is only a fence in the CUDA
    stream the write named. Work on other streams is not ordered after OVRTX's read. Slot refills
    are safe because they run on the same stream that every write names. A refill from any other
    stream would race the read.
    """

    # See :meth:`_create_slots` for why two is always enough.
    _NUM_SLOTS = 2

    # One frame of camera latency. The ring holds one more render because a frame is drained only
    # after the next one is enqueued. Deeper queues are not supported.
    _LATENCY_FRAMES = 1

    @classmethod
    def try_create(cls, cfg: OVRTXRendererCfg) -> _AsyncRenderStrategy | None:
        """Create the strategy when async rendering is enabled. Return ``None`` otherwise.

        :func:`~isaaclab.renderers.resolve_async_rendering_enabled` reads the flag from the
        configuration and the environment variable.
        """
        return cls() if resolve_async_rendering_enabled(cfg) else None

    def __init__(self) -> None:
        super().__init__()
        self._num_envs = 0
        self._render_queue_depth = self._LATENCY_FRAMES + 1
        self._ring: deque[_AsyncRenderEntry] = deque()
        self._slots: list[_AsyncRenderSlot] = []
        self._slot_index = 0
        self._primed = False
        self._current_slot: _AsyncRenderSlot | None = None

    def _has_pending_ops(self) -> bool:
        """Return whether any render op is still queued."""
        return bool(self._ring)

    def _enqueue_render_op(
        self, op: _AsyncRenderOp, render_data: OVRTXRenderData | None, consume_products: _RenderProductConsumer
    ) -> _AsyncRenderEntry:
        """Queue a render op, drain the oldest render when the ring is full, and advance the staging slot."""
        entry = _AsyncRenderEntry(op, render_data, consume_products)
        self._ring.append(entry)
        try:
            if len(self._ring) >= self._render_queue_depth:
                self._try_drain_one()
        finally:
            # Advance even when the drain raises. A caller that catches the error and keeps
            # stepping must not stage the next frame into the slot of the render just submitted.
            if self._slots:
                self._advance_slot()
        return entry

    def initialize(self, num_envs: int) -> None:
        """Reset staging slots and record the camera count for future slot builds.

        See :meth:`_RenderStrategy.initialize`.
        """
        self._reset_slots(num_envs)

    def _reset_slots(self, num_envs: int) -> None:
        """Finish all queued work, then drop the staging slots.

        ``num_envs`` is the camera count for future slot builds. ``0`` means the renderer is
        unbinding.
        """
        # Deliver queued renders rather than dropping them. Each op is its buffer's only keepalive,
        # and a re-initialize must not discard a frame that is still executing. This is a no-op
        # from cleanup(), which drains the ring first.
        self.settle_before_scene_write()
        for slot in self._slots:
            slot.wait_for_writes()
        self._slots.clear()
        self._slot_index = 0
        self._current_slot = None
        self._num_envs = num_envs
        self._primed = False
        self._ring.clear()

    def _create_slots(self) -> None:
        # Two slots suffice at any render depth. The frame being assembled stages into one slot.
        # The other slot still backs the frame in flight. :meth:`_advance_slot` waits out the
        # incoming slot's writes. Those writes were submitted before the render that has just
        # drained, so they are already complete.
        assert self._warp_device is not None
        for _ in range(self._NUM_SLOTS):
            self._slots.append(
                _AsyncRenderSlot(
                    camera_transforms=wp.zeros(self._num_envs, dtype=wp.mat44d, device=self._warp_device),
                    camera_quats=wp.empty(self._num_envs, dtype=wp.quatf, device=self._warp_device),
                    object_transforms=None,
                    write_ops=[],
                )
            )

    def _staging_slot(self) -> _AsyncRenderSlot:
        """The slot that receives this frame's staged transforms, in any staging order.

        The slot pool is built on first use, when the device and camera count are known. After
        that, only :meth:`_advance_slot` rotates slots. Staging calls never rotate them.
        """
        if not self._slots:
            self._create_slots()
            self._current_slot = self._slots[self._slot_index]
        assert self._current_slot is not None
        return self._current_slot

    def _advance_slot(self) -> None:
        """Rotate to the next staging slot. Runs once per frame, when its render is enqueued.

        The incoming slot's writes belong to the frame that was drained just before this call.
        The wait therefore completes immediately in steady state.
        """
        self._slot_index = (self._slot_index + 1) % len(self._slots)
        slot = self._slots[self._slot_index]
        slot.wait_for_writes()
        self._current_slot = slot

    def _write_binding_async(self, slot: _AsyncRenderSlot, binding: Any, data: wp.array) -> None:
        """Record an async binding write on ``slot``, using the device's Warp stream for OVRTX ordering."""
        slot.record_write(binding, data, self._cuda_stream)

    @contextmanager
    def stage_object_transforms(self, binding: Any, num_rows: int, buffer: wp.array) -> Iterator[wp.array]:
        """Stage object transforms into the frame's slot and write them to ``binding`` on exit.

        See :meth:`_RenderStrategy.stage_object_transforms`. ``buffer`` is unused. A pipelined
        frame cannot share one array with the frame still in flight, so the slot provides a
        double-buffered replacement.
        """
        slot = self._staging_slot()
        object_transforms = slot.object_transforms
        if object_transforms is None or object_transforms.shape[0] != num_rows:
            object_transforms = wp.zeros(num_rows, dtype=wp.mat44d, device=self._warp_device)
            slot.object_transforms = object_transforms
        yield object_transforms
        self._write_binding_async(slot, binding, object_transforms)

    @contextmanager
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> Iterator[tuple[wp.array, wp.array]]:
        """Stage camera transforms into the frame's slot and write them to ``binding`` on exit.

        See :meth:`_RenderStrategy.stage_camera_transforms`. Camera and object updates share the
        frame's slot in any order. The camera buffers are reallocated when ``num_rows`` differs
        from their pre-sized ``num_envs``.
        """
        slot = self._staging_slot()
        if slot.camera_transforms.shape[0] != num_rows:
            slot.camera_transforms = wp.zeros(num_rows, dtype=wp.mat44d, device=self._warp_device)
            slot.camera_quats = wp.empty(num_rows, dtype=wp.quatf, device=self._warp_device)
        yield slot.camera_quats, slot.camera_transforms
        self._write_binding_async(slot, binding, slot.camera_transforms)

    def render(
        self,
        renderer: Renderer,
        render_products: set[str],
        delta_time: float,
        render_data: OVRTXRenderData,
        consume_products: _RenderProductConsumer,
        ordinal: int | None = None,
    ) -> None:
        """Start an asynchronous render and queue it for later delivery.

        The first frame of a scene is delivered immediately. The first camera read therefore
        returns a rendered frame instead of the zero-initialized output buffer. Later frames are
        pipelined. See :meth:`_RenderStrategy.render`.
        """
        # The flag marks the first frame. An empty ring cannot: scene writes can drain the ring
        # dry on every frame.
        is_first_frame = not self._primed
        op = renderer.step_async(render_products=render_products, delta_time=delta_time, ordinal=ordinal)
        self._enqueue_render_op(op, render_data, consume_products)
        self._primed = True
        if is_first_frame:
            self._try_drain_one()

    def settle_before_scene_write(self) -> None:
        """Wait for every queued render. See :meth:`_RenderStrategy.settle_before_scene_write`.

        Waiting here, at the next frame's first write, keeps the overlap window open across the
        caller's own work between the frames.
        """
        while self._has_pending_ops():
            self._try_drain_one()

    def _try_drain_one(self) -> bool:
        """Complete the oldest queued render and deliver it. Returns False when nothing was queued."""
        return bool(self._ring) and self._ring.popleft().deliver()

    def release_render_data(self, render_data: OVRTXRenderData) -> None:
        """Stop delivering queued frames into ``render_data``.

        See :meth:`_RenderStrategy.release_render_data`. The queued renders still complete. Their
        delivery then skips the released camera.
        """
        for entry in self._ring:
            if entry.render_data is render_data:
                entry.render_data = None

    def cleanup(self) -> list[Exception]:
        """Finish all queued renders, drop the staging slots, and return the collected failures.

        One bad op must not block the rest of the teardown, so failures are collected instead of
        raised. The caller re-raises them after it has released its backend resources. See
        :meth:`_RenderStrategy.cleanup`.
        """
        errors: list[Exception] = []
        while self._has_pending_ops():
            try:
                self._try_drain_one()
            except Exception as e:
                logger.warning("Error draining OVRTX async render op: %s", e, exc_info=True)
                errors.append(e)

        # Same collect-and-continue rule for the slots' binding writes. ``wait_for_writes`` clears
        # a slot's ops even on failure, so the reset below cannot raise out of the teardown.
        for slot in self._slots:
            try:
                slot.wait_for_writes()
            except Exception as e:
                logger.warning("Error completing OVRTX async binding write during cleanup: %s", e, exc_info=True)
                errors.append(e)
        self._reset_slots(0)
        return errors
