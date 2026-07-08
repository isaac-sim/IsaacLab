# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA-graph capture helper for kernel-only sensor updates."""

from __future__ import annotations

import logging
from collections.abc import Callable

import warp as wp

logger = logging.getLogger(__name__)


class CapturedKernelUpdate:
    """Captures a kernel-only update callable into a CUDA graph and replays it.

    The callable passed to :meth:`run` must contain only warp kernel launches and
    ``wp.copy`` calls on preallocated arrays: no allocations, no host reads, and no
    Python branching on GPU data. Captured array pointers are baked into the graph,
    so all inputs must be pointer-stable buffers that are refreshed in place.

    The graph is captured on the first :meth:`run` and replayed afterwards. On
    non-CUDA devices, or after a capture failure (which logs a warning), the
    callable simply runs eagerly. Call :meth:`invalidate` whenever the buffers the
    callable reads or writes are re-created.
    """

    def __init__(self, device: str, owner: str):
        """Initialize the helper.

        Args:
            device: Device the update kernels run on.
            owner: Human-readable owner description used in error and warning
                messages, e.g. ``"contact sensor at '/World/Robot/foot'"``.
        """
        self._device = wp.get_device(device)
        self._owner = owner
        self.enabled: bool = self._device.is_cuda
        """Whether updates run through a captured CUDA graph. Set to False to force eager launches."""
        self._graph: wp.Graph | None = None

    def refuse_outer_capture(self) -> None:
        """Raises when an outer CUDA graph capture is active on the device.

        Sensor updates that fetch physics state through host-side native calls
        cannot be captured: replays of the outer graph would never re-run the
        fetch and would silently consume frozen data. Call this before any
        fetch work in the update path.

        Raises:
            RuntimeError: If the device is currently capturing.
        """
        if self._device.is_capturing:
            raise RuntimeError(
                f"Cannot update the {self._owner} while a CUDA graph capture is active: the"
                " physics-state fetch cannot be graph-captured, so replaying the captured graph"
                " would consume stale data."
            )

    def run(self, compute: Callable[[], None]) -> None:
        """Runs ``compute`` through the captured graph, capturing it on first use.

        Args:
            compute: Kernel-only callable (see class docstring for restrictions).
        """
        if not self.enabled:
            compute()
            return
        if self._graph is None:
            try:
                with wp.ScopedCapture(device=self._device) as capture:
                    compute()
            except Exception as exc:
                self.enabled = False
                logger.warning(
                    f"Failed to capture the update of the {self._owner} into a CUDA graph."
                    f" Falling back to eager kernel launches. Reason: {exc}"
                )
                compute()
                return
            self._graph = capture.graph
        wp.capture_launch(self._graph)

    def invalidate(self) -> None:
        """Drops the captured graph so the next :meth:`run` re-captures."""
        self._graph = None
