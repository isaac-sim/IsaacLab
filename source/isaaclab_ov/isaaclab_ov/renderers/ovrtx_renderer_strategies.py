# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering execution strategies for the OVRTX renderer.

The renderer delegates *how* a frame is executed -- how transform writes are staged and how the OVRTX
step is dispatched and consumed -- to a :class:`_RenderStrategy`, selected once from configuration so
the renderer's call sites carry no ``sync``/``async`` branching.

- :class:`_SyncRenderStrategy` writes transforms straight into OVRTX and consumes each step inline.
- :class:`_AsyncRenderStrategy` pipelines steps and double-buffers transform staging, so rendering
  overlaps simulation at the cost of camera outputs arriving one frame later.

The interface is mechanism-neutral (:meth:`~_RenderStrategy.initialize`,
:meth:`~_RenderStrategy.stage_object_transforms`, :meth:`~_RenderStrategy.stage_camera_transforms`,
:meth:`~_RenderStrategy.render`, :meth:`~_RenderStrategy.cleanup`); slot and queue vocabulary stays
private to :class:`_AsyncRenderStrategy`.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import warp as wp
from ovrtx import DataAccess, Device

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
        self.products: RenderProductSetOutputs | None = None

    def get_products(self):
        if self.products is None:
            self.products = self.op.wait().fetch()
        return self.products

    def deliver(self) -> bool:
        """Wait for the step, then hand its products to the destination it was submitted for."""
        products = self.get_products()
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
        self._device: str | None = None

    def set_device(self, device: str) -> None:
        """Record the CUDA device used for staging buffers and binding maps."""
        self._device = device

    @property
    def _device_id(self) -> int:
        """CUDA device index parsed from :attr:`_device` for OVRTX ``binding.map()`` calls."""
        assert self._device is not None
        parts = self._device.split(":")
        return int(parts[1]) if len(parts) > 1 else 0

    def initialize(self, num_envs: int) -> None:
        """Prepare per-scene staging resources for ``num_envs`` environments.

        Called once the scene is (re)initialized. The default does nothing; strategies that own
        staging buffers override it.
        """

    def cleanup(self, render_data: OVRTXRenderData | None, consume_products: _RenderProductConsumer) -> None:
        """Flush any in-flight work and release staging resources.

        The default does nothing; strategies that own queued work override it.
        """

    def settle_before_scene_write(self) -> None:
        """Drain in-flight renders that could observe a scene mutation issued after them.

        Backends whose writes land in storage the renderer reads in place call this before mutating the
        scene. The default does nothing; strategies holding in-flight renders override it.
        """

    @abstractmethod
    def stage_object_transforms(self, binding: Any, num_rows: int) -> AbstractContextManager[wp.array]:
        """Yield a ``mat44d`` buffer of ``num_rows`` object transforms for the caller's kernel to fill.

        The yielded buffer is published to ``binding`` when the context exits without error.
        """

    @abstractmethod
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> AbstractContextManager[tuple[wp.array, wp.array]]:
        """Yield ``(quats, transforms)`` staging buffers for ``num_rows`` cameras.

        ``quats`` is a ``quatf`` scratch buffer and ``transforms`` is the ``mat44d`` destination; the
        latter is published to ``binding`` when the context exits without error.
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
    """Blocking strategy: transform writes go straight to OVRTX and each step is consumed inline."""

    @contextmanager
    def stage_object_transforms(self, binding: Any, num_rows: int) -> Iterator[wp.array]:
        """Map ``binding`` and yield its buffer so the caller's kernel writes OVRTX storage directly.

        See :meth:`_RenderStrategy.stage_object_transforms`. ``num_rows`` is implied by the binding.
        """
        with binding.map(device=Device.CUDA, device_id=self._device_id) as attr_mapping:
            yield wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)

    @contextmanager
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> Iterator[tuple[wp.array, wp.array]]:
        """Yield freshly allocated staging buffers and copy the result into ``binding`` on exit.

        See :meth:`_RenderStrategy.stage_camera_transforms`.
        """
        camera_quats = wp.empty(num_rows, dtype=wp.quatf, device=self._device)
        camera_transforms = wp.zeros(num_rows, dtype=wp.mat44d, device=self._device)
        yield camera_quats, camera_transforms
        with binding.map(device=Device.CUDA, device_id=self._device_id) as attr_mapping:
            wp_transforms_view = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
            wp.copy(wp_transforms_view, camera_transforms)

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

        ``cuda_stream`` is the Warp stream the staging kernel that filled ``data`` ran on. Handing it to
        OVRTX lets OVRTX insert a GPU-side wait before its ``DataAccess.ASYNC`` read, so the subsequent
        ``step_async`` never observes a partially written transform buffer. Without it OVRTX performs no
        cross-stream sync against the fill kernel and the read races the write.
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

    Transform writes always use two slots. A ``DataAccess.ASYNC`` write copies the data into
    OVRTX's own storage, so OVRTX reads a buffer only until its write op completes. A buffer is safe
    to recycle once its write has drained.
    """

    # See :meth:`_ensure_slots` for why two is always enough.
    _NUM_SLOTS = 2
    # Renders kept in flight before the oldest is drained. Two means one frame of camera latency.
    # Overridable via ``OVRTX_NUM_BUFFERS`` (see :meth:`_resolve_render_queue_depth`).
    _DEFAULT_RENDER_QUEUE_DEPTH = 2

    @classmethod
    def try_create(cls, cfg: OVRTXRendererCfg) -> _AsyncRenderStrategy | None:
        """Create an :class:`_AsyncRenderStrategy` when async rendering is enabled, else return ``None``.

        Async rendering is toggled by :attr:`OVRTXRendererCfg.async_rendering`. The
        ``OVRTX_ASYNC_RENDERING`` environment variable overrides the config (``0``/``false``/``no``/``off``
        disable, any other non-empty value enables) so existing golden-image tests can exercise the
        async path without editing task configs.
        """
        enabled = bool(getattr(cfg, "async_rendering", False))
        env_override = os.environ.get("OVRTX_ASYNC_RENDERING")
        if env_override is not None and env_override != "":
            enabled = env_override.strip().lower() not in ("0", "false", "no", "off")
        return cls() if enabled else None

    @classmethod
    def _resolve_render_queue_depth(cls) -> int:
        """Return the render queue depth, honoring the ``OVRTX_NUM_BUFFERS`` environment variable.

        The value is the number of ``step_async`` renders kept in flight before the oldest is drained;
        larger values overlap more simulation with rendering at the cost of extra frames of camera
        latency. Falls back to :attr:`_DEFAULT_RENDER_QUEUE_DEPTH` when unset; a value below 2 would
        disable pipelining and is clamped up to 2.
        """
        raw = os.environ.get("OVRTX_NUM_BUFFERS")
        if raw is None or raw.strip() == "":
            return cls._DEFAULT_RENDER_QUEUE_DEPTH
        try:
            value = int(raw.strip())
        except ValueError:
            logger.warning(
                "Ignoring invalid OVRTX_NUM_BUFFERS=%r; using default %d.", raw, cls._DEFAULT_RENDER_QUEUE_DEPTH
            )
            return cls._DEFAULT_RENDER_QUEUE_DEPTH
        if value < 2:
            logger.warning("OVRTX_NUM_BUFFERS=%d is below the minimum of 2; clamping to 2.", value)
            return 2
        return value

    def __init__(self) -> None:
        super().__init__()
        self._num_envs = 0
        self._render_queue_depth = self._resolve_render_queue_depth()
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
        """Record an async render op and end the current frame's slot, draining one slot when the ring is full."""
        entry = _AsyncRenderEntry(op, render_data, consume_products)
        self._ring.append(entry)
        self._current_slot = None
        if len(self._ring) >= self._render_queue_depth:
            self._try_drain_one()
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
        # An empty ring marks the first frame in :meth:`render`, which primes it synchronously.
        self._ring.clear()

    def _ensure_slots(self) -> None:
        if self._slots:
            return

        # Two slots suffice at any render depth: with one write per frame, a slot's only outstanding
        # work on reuse is its write from two frames ago, which :meth:`_begin_slot` waits on. OVRTX
        # copies the buffer into its own storage before that write op completes.
        assert self._device is not None
        for _ in range(self._NUM_SLOTS):
            self._slots.append(
                _AsyncRenderSlot(
                    camera_transforms=wp.zeros(self._num_envs, dtype=wp.mat44d, device=self._device),
                    camera_quats=wp.empty(self._num_envs, dtype=wp.quatf, device=self._device),
                    object_transforms=None,
                    write_ops=[],
                )
            )

    def _begin_slot(self) -> _AsyncRenderSlot:
        """Rotate to the next staging slot for a new frame's transform writes."""
        self._ensure_slots()
        slot = self._slots[self._slot_index]
        self._slot_index = (self._slot_index + 1) % len(self._slots)
        slot.wait_for_writes()
        self._current_slot = slot
        return slot

    def _current_or_begin_slot(self) -> _AsyncRenderSlot:
        """Return the slot for the current frame, starting one if none is active yet."""
        return self._current_slot or self._begin_slot()

    def _write_binding_async(self, slot: _AsyncRenderSlot, binding: Any, data: wp.array) -> None:
        """Record an async binding write on ``slot``, using the device's Warp stream for OVRTX ordering."""
        slot.record_write(binding, data, wp.get_stream(self._device).cuda_stream)

    @contextmanager
    def stage_object_transforms(self, binding: Any, num_rows: int) -> Iterator[wp.array]:
        """Stage object transforms into a fresh frame slot; publish them on exit.

        See :meth:`_RenderStrategy.stage_object_transforms`. Uses :meth:`_begin_slot`, so each object
        update opens (and double-buffers) a new slot for the frame.
        """
        slot = self._begin_slot()
        object_transforms = slot.object_transforms
        if object_transforms is None or object_transforms.shape[0] != num_rows:
            object_transforms = wp.zeros(num_rows, dtype=wp.mat44d, device=self._device)
            slot.object_transforms = object_transforms
        yield object_transforms
        self._write_binding_async(slot, binding, object_transforms)

    @contextmanager
    def stage_camera_transforms(self, binding: Any, num_rows: int) -> Iterator[tuple[wp.array, wp.array]]:
        """Stage camera transforms into the current frame slot; publish them on exit.

        See :meth:`_RenderStrategy.stage_camera_transforms`. Uses :meth:`_current_or_begin_slot` to join
        the slot opened by the object update (or open one when there was no object update this frame).
        The slot's camera buffers are pre-sized to ``num_envs`` in :meth:`_ensure_slots`; reallocate
        them if ``num_rows`` diverges, mirroring :meth:`stage_object_transforms`.
        """
        slot = self._current_or_begin_slot()
        if slot.camera_transforms.shape[0] != num_rows:
            slot.camera_transforms = wp.zeros(num_rows, dtype=wp.mat44d, device=self._device)
            slot.camera_quats = wp.empty(num_rows, dtype=wp.quatf, device=self._device)
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

        The first frame of a scene is primed synchronously: the op is waited on and consumed
        immediately so the first read returns a rendered frame rather than the zero-initialized output
        buffer. Priming is tracked explicitly rather than inferred from an empty ring, because a
        backend that drains the ring on every scene write leaves it empty at every render. The entry
        caches its products, so the next frame's drain reuses them instead of fetching twice. Later
        frames are pipelined; a drain only replaces buffer contents, so the output stays valid while
        the queue fills. See :meth:`_RenderStrategy.render`.
        """
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

    def cleanup(
        self,
        render_data: OVRTXRenderData | None,
        consume_products: _RenderProductConsumer,
    ) -> None:
        """Drain all queued renders best-effort and drop staging slots.

        See :meth:`_RenderStrategy.cleanup`.
        """
        # Log and continue: a failure on one op must not prevent draining the rest or tearing the
        # renderer down. The per-frame drain in :meth:`_enqueue_render_op` propagates instead.
        while self._has_pending_ops():
            try:
                self._try_drain_one()
            except Exception as e:
                logger.warning("Error draining OVRTX async render op: %s", e, exc_info=True)

        self._reset_slots(0)
