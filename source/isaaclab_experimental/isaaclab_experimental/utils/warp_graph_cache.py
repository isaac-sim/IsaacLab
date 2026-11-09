# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp CUDA graph capture-or-replay utility."""

from collections.abc import Callable
from typing import Any

import warp as wp


class WarpGraphCache:
    """Caches Warp CUDA graphs by stage name: captures on first call, replays after.

    On the very first call for a given stage, an **eager warm-up** run
    executes *before* graph capture.  This lets one-time initialisation
    code (memory allocations, torch dtype casts, ``hasattr`` guards, etc.)
    run outside the capture context.  Only the steady-state kernel
    launches are then recorded into the graph.

    The return value from the capture run is cached and returned on every
    subsequent replay, ensuring captured stages return the same references
    (e.g. tensor views) as eager stages.

    Usage::

        cache = WarpGraphCache()
        result = cache.capture_or_replay("my_stage", my_warp_function)
        # uncaptured work here ...
        result2 = cache.capture_or_replay("my_stage_post", my_other_function)
    """

    def __init__(self):
        self._graphs: dict[str, Any] = {}
        self._results: dict[str, Any] = {}

    def capture_or_replay(
        self,
        stage: str,
        fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Capture *fn* into a CUDA graph on the first call, then replay.

        Args:
            stage: Unique name identifying this captured scope.
            fn: The callable to capture. Must contain only CUDA-graph-safe
                operations (pure warp kernels, no Python-level branching on
                GPU data).
            args: Positional arguments forwarded to *fn*. Defaults to ``()``.
            kwargs: Keyword arguments forwarded to *fn*. Defaults to ``None``.

        Returns:
            The cached return value from the first (capture) invocation.
        """
        if kwargs is None:
            kwargs = {}
        graph = self._graphs.get(stage)
        if graph is not None:
            wp.capture_launch(graph)
            return self._results[stage]
        # Warm-up: run eagerly to flush first-call allocations / hasattr guards.
        fn(*args, **kwargs)
        # Capture: allocations already done, only wp.launch calls are recorded.
        with wp.ScopedCapture() as capture:
            result = fn(*args, **kwargs)
        self._graphs[stage] = capture.graph
        self._results[stage] = result
        return result

    def invalidate(self, stage: str | None = None) -> None:
        """Drop cached graph(s). If *stage* is ``None``, drop all."""
        if stage is None:
            self._graphs.clear()
            self._results.clear()
        else:
            self._graphs.pop(stage, None)
            self._results.pop(stage, None)
