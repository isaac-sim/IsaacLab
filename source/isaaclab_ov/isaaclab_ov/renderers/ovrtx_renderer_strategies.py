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
    """An in-flight render step together with the destination its products belong to.

    The entry carries its own ``render_data`` and consumer so a drain triggered from an unrelated call
    site -- a scene write, or teardown -- still delivers the frame to the buffers the step was
    submitted for.
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
        """Wait for the step, then hand its products to the destination it was submitted for."""
        products = self.op.wait().fetch()
        if products is None:
            return False
        if self.render_data is not None:
            self.consume_products(self.render_data, products)
        return True


class _RenderStrategy(ABC):
    """Strategy for how the OVRTX renderer stages transforms and dispatches render steps.

    Two concrete strategies exist: :class:`_SyncRenderStrategy` steps OVRTX and consumes its products
    inline, while :class:`_AsyncRenderStrategy` pipelines steps and double-buffers transform staging.
    The renderer drives whichever it holds through this neutral interface, so its call sites carry no
    ``sync``/``async`` branching. Slot/queue vocabulary stays private to :class:`_AsyncRenderStrategy`.
    """

    def __init__(self) -> None:
        self._warp_device: wp.Device | None = None

    def set_device(self, warp_device: wp.Device) -> None:
        """Record the renderer's resolved Warp device used for staging buffers and sync streams."""
        self._warp_device = warp_device

    @property
    def _cuda_stream(self) -> int:
        """The device's current Warp CUDA stream, which OVRTX orders its reads against.

        Read at use time from the cached device, never cached itself: the current stream can
        legitimately change (e.g. CUDA graph capture), the device cannot.
        """
        return self._warp_device.stream.cuda_stream

    def initialize(self, num_envs: int) -> None:
        """Prepare per-scene staging resources for ``num_envs`` environments.

        Called once the scene is (re)initialized. The default does nothing; strategies that own
        staging buffers override it.
        """

    def cleanup(self) -> None:
        """Flush in-flight work and release staging resources. The default does nothing."""

    def settle_before_scene_write(self) -> None:
        """Drain in-flight renders that could observe a scene mutation issued after them.

        Backends whose writes land in storage the renderer reads in place call this before mutating the
        scene. The default does nothing; strategies holding in-flight renders override it.
        """

    @abstractmethod
    def stage_object_transforms(
        self, binding: Any, num_rows: int, buffer: wp.array
    ) -> AbstractContextManager[wp.array]:
        """Yield a ``mat44d`` buffer of ``num_rows`` object transforms for the caller's kernel to fill.

        ``buffer`` is the caller's persistent staging array; a strategy that owns its own buffers may
        yield one of those instead. The yielded buffer is published to ``binding`` when the context
        exits without error. The caller must enqueue its fill work on the device's current Warp stream
        (the default for ``wp.launch``/``wp.copy``): publication orders OVRTX's read of the buffer
        against that stream, so the fill needs no explicit stream handoff of its own.
        """

    @abstractmethod
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> AbstractContextManager[tuple[wp.array, wp.array]]:
        """Yield ``(quats, transforms)`` staging buffers for ``num_rows`` cameras.

        ``quats`` is a ``quatf`` scratch buffer and ``transforms`` is the ``mat44d`` destination; the
        latter is published to ``binding`` when the context exits without error. The same
        current-Warp-stream contract as :meth:`stage_object_transforms` applies.
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
        """Step ``renderer`` for one frame and consume its products, immediately or deferred.

        ``ordinal`` is the minimum committed ovstage publication the step must observe. It is required
        while an ovstage is attached to ``renderer`` and rejected otherwise, so the caller passes the
        value matching its scene-ownership path.
        """


class _SyncRenderStrategy(_RenderStrategy):
    """Blocking strategy: transform writes go straight to OVRTX and each step is consumed inline.

    Publication uses a blocking ``write()``, so the staged buffer stays valid until OVRTX has read it.
    ``DataAccess.ASYNC`` plus the producing Warp stream lets OVRTX read in place and wait on-GPU for
    the fill kernel; ``SYNC`` is rejected for GPU buffers.
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
        """Issue an async binding write and record its op so it can be drained before reuse.

        ``cuda_stream`` is the Warp stream the staging kernel that filled ``data`` ran on. OVRTX
        synchronizes with it before reading ``data`` (today a host-side wait on its operation
        thread, off this Python thread), so the ``DataAccess.ASYNC`` read never observes a partially
        written transform buffer. Its API contract requires this handoff for GPU data; without it,
        ordering would rest on OVRTX committing via the legacy default CUDA stream, an
        implementation detail that happens to serialize with Warp's blocking streams today.
        """
        self.write_ops.append(binding.write_async(data, data_access=DataAccess.ASYNC, cuda_stream=cuda_stream))

    def wait_for_writes(self) -> None:
        """Block until this slot's outstanding async writes drain, so its buffers are safe to reuse."""
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
    """Pipelined strategy: steps are queued and consumed later, with double-buffered staging.

    The queue sustains one frame of camera latency when
    :attr:`~isaaclab.renderers.RendererCfg.async_rendering` is enabled, so rendering overlaps the
    next step's simulation and Python work and camera outputs are one step stale.

    Transform writes always use two slots, independent of queue depth. A ``DataAccess.ASYNC`` write
    copies the data into OVRTX's own storage, so OVRTX reads a buffer only until its write op
    completes. A buffer is safe to recycle once its write has drained.
    """

    # See :meth:`_create_slots` for why two is always enough.
    _NUM_SLOTS = 2

    # One frame of camera latency; the ring holds one more render because a frame is drained only
    # after the next one is enqueued. Deeper queues are deliberately unsupported for now: the
    # ovstage path cannot sustain them (its scene writes drain in-flight renders), and the legacy
    # path has not measured a benefit that would justify the extra review surface.
    _LATENCY_FRAMES = 1

    @classmethod
    def try_create(cls, cfg: OVRTXRendererCfg) -> _AsyncRenderStrategy | None:
        """Create an :class:`_AsyncRenderStrategy` when async rendering is enabled, else return ``None``.

        The flag comes from :attr:`~isaaclab.renderers.RendererCfg.async_rendering`, with
        :data:`~isaaclab.renderers.ASYNC_RENDERING_ENV_VAR` taking precedence so golden-image tests
        can exercise the async path without editing task configs.
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
        if len(self._ring) >= self._render_queue_depth:
            self._try_drain_one()
        if self._slots:
            self._advance_slot()
        return entry

    def initialize(self, num_envs: int) -> None:
        """Reset staging slots and record the camera count for future slot builds.

        See :meth:`_RenderStrategy.initialize`.
        """
        self._reset_slots(num_envs)

    def _reset_slots(self, num_envs: int) -> None:
        """Drain and drop all staging slots, and record the camera count for future slot builds.

        ``num_envs == 0`` means the renderer is unbinding, so slots are simply cleared.
        """
        for slot in self._slots:
            slot.wait_for_writes()
        self._slots.clear()
        self._slot_index = 0
        self._current_slot = None
        self._num_envs = num_envs
        self._primed = False
        self._ring.clear()

    def _create_slots(self) -> None:
        # Two slots suffice at any render depth: the frame being assembled stages into one slot
        # while the other still backs the frame in flight. :meth:`_advance_slot` waits out the
        # incoming slot's writes, which were submitted before the render that has just drained and
        # are therefore already complete. OVRTX copies the buffer into its own storage before that
        # write op completes.
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
        """The slot receiving this frame's staged transforms, whatever order the stagers run in.

        The slot pool is built on first use (device and camera count are known by then). After
        that, the only lifecycle event is :meth:`_advance_slot` when the frame's render is
        enqueued; staging calls never rotate slots themselves.
        """
        if not self._slots:
            self._create_slots()
            self._current_slot = self._slots[self._slot_index]
        assert self._current_slot is not None
        return self._current_slot

    def _advance_slot(self) -> None:
        """Rotate to the next staging slot, called once per frame when its render is enqueued.

        The incoming slot's outstanding writes belong to the frame whose render was drained just
        before this call, so the wait completes immediately in steady state; it can only block on
        a write op that outlived its own render.
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
        """Stage object transforms into the frame's slot; publish them on exit.

        See :meth:`_RenderStrategy.stage_object_transforms`. ``buffer`` is unused because a
        pipelined frame cannot share one array with the frame still in flight; the slot provides
        the double-buffered replacement.
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
        """Stage camera transforms into the frame's slot; publish them on exit.

        See :meth:`_RenderStrategy.stage_camera_transforms`. Camera and object updates share the
        frame's slot in whichever order the frame runs them. The slot's camera buffers are
        reallocated when ``num_rows`` diverges from their pre-sized ``num_envs``.
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
        """Step OVRTX asynchronously and enqueue the op for deferred consumption.

        The first frame of a scene is drained immediately, so the first camera read returns a rendered
        frame rather than the zero-initialized output buffer. Later frames are pipelined; a drain only
        replaces buffer contents, so the output stays valid while the queue fills.
        See :meth:`_RenderStrategy.render`.
        """
        # The flag, not an empty ring, marks the first frame: scene writes can drain the ring dry every frame.
        is_first_frame = not self._primed
        op = renderer.step_async(render_products=render_products, delta_time=delta_time, ordinal=ordinal)
        self._enqueue_render_op(op, render_data, consume_products)
        self._primed = True
        if is_first_frame:
            self._try_drain_one()

    def settle_before_scene_write(self) -> None:
        """Drain every queued render so the caller's scene write cannot alter a frame in flight.

        See :meth:`_RenderStrategy.settle_before_scene_write`. Draining here keeps the pipelining
        window open across the caller's own work -- simulation, inference and product reads all
        overlap the render, and only the next frame's first write closes it.
        """
        while self._has_pending_ops():
            self._try_drain_one()

    def _try_drain_one(self) -> bool:
        """Complete the oldest queued render and deliver it. Returns False when nothing was queued."""
        return bool(self._ring) and self._ring.popleft().deliver()

    def cleanup(self) -> None:
        """Drain all queued renders best-effort and drop staging slots.

        Each render delivers into the buffers it was submitted with. See :meth:`_RenderStrategy.cleanup`.
        """
        # Log and continue: one bad op must not block draining the rest or tearing the renderer down.
        while self._has_pending_ops():
            try:
                self._try_drain_one()
            except Exception as e:
                logger.warning("Error draining OVRTX async render op: %s", e, exc_info=True)

        # Same best-effort rule for the slots' binding writes: ``wait_for_writes`` clears a slot's
        # ops even on failure, so the reset below cannot raise out of the renderer's teardown.
        for slot in self._slots:
            try:
                slot.wait_for_writes()
            except Exception as e:
                logger.warning("Error completing OVRTX async binding write during cleanup: %s", e, exc_info=True)
        self._reset_slots(0)
