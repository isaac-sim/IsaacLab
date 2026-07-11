# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CUDA graph for Newton shape-BVH refit and its consumers."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)


class BvhTaskGraph:
    """One CUDA graph shared by every consumer of the Newton shape BVH.

    Consumers of the scene BVH (tiled-camera renderers, ray-cast sensors)
    register a graph-capturable launch function under a unique name. All
    registered tasks are captured into a single CUDA graph together with the
    BVH refit, each gated by a device-side condition flag
    (:func:`warp.capture_if`). A :meth:`run` call launches that one graph with
    only the requested tasks' flags raised; the refit flag is raised only when
    the simulation state changed since the last refit
    (see :meth:`mark_state_changed`).

    This resolves consumers updating at independent frequencies: whichever
    consumer runs first after a physics step pays for the refit, later
    consumers in the same step reuse it, and idle consumers cost nothing.

    When CUDA graphs are unavailable (CPU device, capture failure, or disabled
    in the config) the same flag logic runs eagerly on the host.
    """

    def __init__(
        self,
        refit_fn: Callable[[], None],
        capture_fn: Callable[[Callable[[], None]], wp.Graph | None],
        device: str,
        use_cuda_graph: bool,
    ):
        """Initialize the task graph.

        Args:
            refit_fn: Graph-capturable function that refits the shape BVH for
                the current simulation state.
            capture_fn: Strategy that captures a callable into a CUDA graph and
                returns it, or ``None`` on failure (triggers eager fallback).
            device: Device the tasks launch on.
            use_cuda_graph: Whether to capture a CUDA graph at all.
        """
        self._refit_fn = refit_fn
        self._capture_fn = capture_fn
        self._device = device
        self._use_cuda_graph = use_cuda_graph and "cuda" in device
        self._tasks: dict[str, Callable[[], None]] = {}
        self._graph: wp.Graph | None = None
        self._flags: wp.array | None = None
        self._flags_host: np.ndarray | None = None
        self._state_dirty = True

    def register(self, name: str, fn: Callable[[], None]) -> None:
        """Register a graph-capturable task that consumes the refit BVH.

        Registration invalidates a previously captured graph; the next
        :meth:`run` re-captures with the new task included.

        Args:
            name: Unique task name.
            fn: Function that only issues graph-capturable work (kernel
                launches on stable buffers, no allocations or host syncs).
        """
        if name in self._tasks:
            raise ValueError(f"BVH task '{name}' is already registered.")
        self._tasks[name] = fn
        self._graph = None

    def unregister(self, name: str) -> None:
        """Remove a task; a no-op when the name is unknown."""
        if self._tasks.pop(name, None) is not None:
            self._graph = None

    def mark_state_changed(self) -> None:
        """Flag that body poses changed, so the next :meth:`run` refits the BVH."""
        self._state_dirty = True

    def run(self, *names: str) -> None:
        """Refit the BVH (if the state changed) and execute the named tasks.

        Args:
            names: Names of previously registered tasks to execute.
        """
        for name in names:
            if name not in self._tasks:
                raise KeyError(f"BVH task '{name}' is not registered.")
        if self._use_cuda_graph and self._graph is None:
            self._capture()
        if self._graph is None:
            if self._state_dirty:
                self._refit_fn()
                self._state_dirty = False
            for name in names:
                self._tasks[name]()
            return

        self._flags_host[:] = 0
        self._flags_host[0] = 1 if self._state_dirty else 0
        task_names = list(self._tasks)
        for name in names:
            self._flags_host[1 + task_names.index(name)] = 1
        self._flags.assign(self._flags_host)
        wp.capture_launch(self._graph)
        self._state_dirty = False

    def _capture(self) -> None:
        """Capture refit + all tasks into one conditional CUDA graph."""
        # Warmup: compile kernels and pin scratch allocations before capture.
        with wp.ScopedDevice(self._device):
            self._refit_fn()
            for fn in self._tasks.values():
                fn()

        self._flags = wp.zeros(1 + len(self._tasks), dtype=wp.int32, device=self._device)
        self._flags_host = np.zeros(1 + len(self._tasks), dtype=np.int32)
        task_fns = list(self._tasks.values())

        def pipeline():
            wp.capture_if(self._flags[0:1], self._refit_fn)
            for i, fn in enumerate(task_fns):
                wp.capture_if(self._flags[i + 1 : i + 2], fn)

        self._graph = self._capture_fn(pipeline)
        if self._graph is None:
            self._use_cuda_graph = False
            logger.warning("BVH task graph capture failed; falling back to eager execution.")
        else:
            logger.info("BVH task graph captured with %d task(s).", len(self._tasks))
