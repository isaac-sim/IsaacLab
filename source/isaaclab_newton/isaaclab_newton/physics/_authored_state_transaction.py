# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-backed transaction for task-authored Newton state writes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum, auto

import warp as wp

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _mark_from_mask(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=int),
    articulation_selection: wp.array(dtype=int),
    use_selection: int,
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
    pending: wp.array(dtype=wp.int32),
):
    """Accumulate selected worlds and their articulations."""
    world = wp.tid()
    count = articulation_selection.shape[0] if use_selection else articulation_ids.shape[1]
    if env_mask[world] and count > 0:
        world_mask[world] = True
        for index in range(count):
            column = articulation_selection[index] if use_selection else index
            fk_mask[articulation_ids[world, column]] = True
        wp.atomic_max(pending, 0, 1)


@wp.kernel(enable_backward=False)
def _mark_from_ids(
    env_ids: wp.array(dtype=int),
    articulation_ids: wp.array2d(dtype=int),
    articulation_selection: wp.array(dtype=int),
    use_selection: int,
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
    pending: wp.array(dtype=wp.int32),
):
    """Accumulate sparse world indices and their articulations."""
    index = wp.tid()
    count = articulation_selection.shape[0] if use_selection else articulation_ids.shape[1]
    if count > 0:
        world = env_ids[index]
        world_mask[world] = True
        for selection_index in range(count):
            column = articulation_selection[selection_index] if use_selection else selection_index
            fk_mask[articulation_ids[world, column]] = True
        wp.atomic_max(pending, 0, 1)


class _CaptureState(Enum):
    """Conditional-graph lifecycle."""

    DISABLED = auto()
    DEFERRED = auto()
    READY = auto()


class AuthoredStateTransaction:
    """Coalesce authored state writes and reconcile them at coherent boundaries.

    The device scalar is authoritative so marking kernels remain valid when
    replayed from an outer CUDA graph. The bound ``apply`` callback receives the
    accumulated world and articulation masks and is responsible for FK and
    solver-specific synchronization.
    """

    RENDER_TRANSFORMS = 1

    def __init__(
        self,
        world_count: int,
        articulation_count: int,
        device,
        apply: Callable[[wp.array, wp.array], None],
    ) -> None:
        self._device = wp.get_device(device)
        self._apply = apply
        self._world_mask = wp.zeros(world_count, dtype=wp.bool, device=self._device)
        self._fk_mask = wp.zeros(articulation_count, dtype=wp.bool, device=self._device)
        self._empty_selection = wp.empty(0, dtype=wp.int32, device=self._device)
        self._pending = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._graph = None
        self._graph_safe = False
        self._capture_state = _CaptureState.DISABLED
        self._writes_may_replay = False
        self._host_pending = False
        self._replay_render_domains = 0

    @property
    def world_mask(self) -> wp.array:
        """Accumulated per-world mask, exposed for private tests and policies."""
        return self._world_mask

    @property
    def fk_mask(self) -> wp.array:
        """Accumulated per-articulation FK mask, exposed for private tests."""
        return self._fk_mask

    @property
    def needs_capture(self) -> bool:
        """Whether a deferred conditional graph still needs capturing."""
        return self._capture_state is _CaptureState.DEFERRED

    @property
    def writes_may_replay(self) -> bool:
        """Whether any mark operation was recorded into a CUDA graph."""
        return self._writes_may_replay

    @property
    def replay_render_domains(self) -> int:
        """Render domains whose host dirty markers may replay only on device."""
        return self._replay_render_domains

    def note_render_write(self, domain: int) -> None:
        """Remember render invalidation recorded while a CUDA stream is capturing."""
        if self._device.is_cuda and self._device.stream.is_capturing:
            self._replay_render_domains |= domain

    def mark_rigid(
        self,
        *,
        env_mask: wp.array | None = None,
        env_ids: wp.array | None = None,
        articulation_ids: wp.array | None = None,
        articulation_selection: wp.array | None = None,
    ) -> None:
        """Accumulate rigid worlds and articulations modified by an asset write."""
        self._host_pending = True
        if self._device.is_cuda and self._device.stream.is_capturing:
            self._writes_may_replay = True

        # Preserve the pre-transaction conservative fallback: without view
        # topology, a rigid write cannot safely map a world selection to
        # articulations, so reconcile the whole model.
        if articulation_ids is None:
            self._world_mask.fill_(True)
            self._fk_mask.fill_(True)
            self._pending.fill_(1)
            return

        if articulation_ids is not None and env_mask is not None:
            selection = self._empty_selection if articulation_selection is None else articulation_selection
            wp.launch(
                _mark_from_mask,
                dim=articulation_ids.shape[0],
                inputs=[env_mask, articulation_ids, selection, int(articulation_selection is not None)],
                outputs=[self._world_mask, self._fk_mask, self._pending],
                device=self._device,
            )
        elif articulation_ids is not None and env_ids is not None:
            selection = self._empty_selection if articulation_selection is None else articulation_selection
            wp.launch(
                _mark_from_ids,
                dim=env_ids.shape[0],
                inputs=[env_ids, articulation_ids, selection, int(articulation_selection is not None)],
                outputs=[self._world_mask, self._fk_mask, self._pending],
                device=self._device,
            )
        else:
            # No selection means every world and articulation.
            self._world_mask.fill_(True)
            self._fk_mask.fill_(True)
            self._pending.fill_(1)

    def configure_capture(self, graph_safe: bool, defer: bool = False, enabled: bool = True) -> None:
        """Configure the reusable conditional graph for the active backend."""
        self._graph = None
        self._graph_safe = graph_safe
        self._capture_state = _CaptureState.DISABLED

        if not enabled or not graph_safe or not self._device.is_cuda:
            return
        if not wp.is_conditional_graph_supported():
            logger.warning(
                "CUDA conditional graph nodes are unavailable; authored-state synchronization will use "
                "a synchronous condition readback outside graph capture."
            )
            return
        if defer:
            self._capture_state = _CaptureState.DEFERRED
            return

        with wp.ScopedCapture(device=self._device) as capture:
            self._capture_condition()
        self._graph = capture.graph
        self._capture_state = _CaptureState.READY

    def capture_deferred(
        self,
        capture_fn: Callable[[Callable[[], None]], object | None],
    ) -> None:
        """Capture a deferred conditional through an RTX-safe capture function."""
        if not self.needs_capture:
            return

        graph = capture_fn(self._capture_condition)
        if graph is None:
            self._capture_state = _CaptureState.DISABLED
            logger.warning("Authored-state graph capture failed; using eager conditional execution.")
            return

        self._graph = graph
        self._capture_state = _CaptureState.READY

    def flush(self) -> None:
        """Apply and consume pending writes, preserving outer-capture replay."""
        capturing = self._device.is_cuda and self._device.stream.is_capturing

        if self._capture_state is _CaptureState.READY and not capturing:
            wp.capture_launch(self._graph)
            self._host_pending = False
            return

        if capturing:
            if not self._graph_safe:
                raise RuntimeError(
                    "Authored-state synchronization for this Newton backend is not CUDA-graph-safe. "
                    "Run the synchronization boundary outside graph capture."
                )
            if not wp.is_conditional_graph_supported():
                raise RuntimeError(
                    "Replaying authored-state writes inside a CUDA graph requires CUDA conditional graph nodes."
                )

        # For eager writers, the host call is an exact clean/dirty hint and
        # avoids synchronizing a CUDA condition at every clean boundary. Once
        # a writer itself was captured, only the device scalar is authoritative.
        if not capturing and not self._host_pending and not self._writes_may_replay:
            return

        # Without an active capture, capture_if directly evaluates the CPU
        # condition or performs the documented CUDA readback fallback.
        wp.capture_if(self._pending, self._consume)
        self._host_pending = False

    def _capture_condition(self) -> None:
        """Insert the device-side transaction branch into the active capture."""
        wp.capture_if(self._pending, self._consume)

    def _consume(self) -> None:
        """Apply the transaction and clear it only after successful dispatch."""
        self._apply(self._world_mask, self._fk_mask)
        self._world_mask.zero_()
        self._fk_mask.zero_()
        self._pending.zero_()
