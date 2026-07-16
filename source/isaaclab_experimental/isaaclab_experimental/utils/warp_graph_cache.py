# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp CUDA graph capture-or-replay utility."""

import os
from collections.abc import Callable
from typing import Any

import warp as wp

_WARP_GRAPH_MODE_ENV = "ISAACLAB_WARP_GRAPH_MODE"


class WarpGraphCache:
    """Cache Warp CUDA graphs by stage name, or execute stages eagerly.

    In capture mode, the first call for a stage executes one eager warm-up
    before graph capture. This lets one-time initialization code (memory
    allocations, Torch dtype casts, ``hasattr`` guards, and similar work)
    run outside the capture context. Only the steady-state kernel
    launches are then recorded into the graph.

    The return value from the capture run is cached and returned on every later
    replay, ensuring captured stages return the same references, such as tensor
    views. In eager mode, every call executes the callable and returns that
    invocation's result without caching.

    Usage::

        cache = WarpGraphCache()
        result = cache.capture_or_replay("my_stage", my_warp_function)
        # uncaptured work here ...
        result2 = cache.capture_or_replay("my_stage_post", my_other_function)

    Args:
        mode: Execution mode. ``"capture"`` caches CUDA graphs and ``"eager"``
            calls each stage directly. If omitted, the value is read from
            ``ISAACLAB_WARP_GRAPH_MODE`` and defaults to ``"capture"``.
    """

    def __init__(self, mode: str | None = None):
        if mode is None:
            mode = os.environ.get(_WARP_GRAPH_MODE_ENV, "capture")
        mode = mode.lower()
        if mode not in {"capture", "eager"}:
            raise ValueError(f"Invalid Warp graph mode {mode!r}. Set {_WARP_GRAPH_MODE_ENV} to 'capture' or 'eager'.")
        self._mode = mode
        self._graphs: dict[str, Any] = {}
        self._results: dict[str, Any] = {}

    def capture_or_replay(
        self,
        stage: str,
        fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a callable eagerly or capture it once and replay its graph.

        Args:
            stage: Unique name identifying this captured scope.
            fn: The callable to capture. Must contain only CUDA-graph-safe
                operations (pure warp kernels, no Python-level branching on
                GPU data).
            args: Positional arguments forwarded to *fn*. Defaults to ``()``.
            kwargs: Keyword arguments forwarded to *fn*. Defaults to ``None``.

        Returns:
            In eager mode, the current invocation's return value. In capture
            mode, the cached return value from the capture invocation.
        """
        if kwargs is None:
            kwargs = {}
        if self._mode == "eager":
            return fn(*args, **kwargs)
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
        """Drain and drop one captured graph or every captured graph.

        Args:
            stage: Stage to invalidate. If ``None``, invalidate every stage.

        Devices are synchronized only when the selected stage owns a captured
        graph. Eager and otherwise empty caches return without synchronizing.
        """
        if stage is None:
            graphs = tuple(self._graphs.values())
            self._synchronize_graphs(graphs)
            self._graphs.clear()
            self._results.clear()
        else:
            graph = self._graphs.get(stage)
            graphs = () if graph is None else (graph,)
            self._synchronize_graphs(graphs)
            self._graphs.pop(stage, None)
            self._results.pop(stage, None)

    @staticmethod
    def _synchronize_graphs(graphs: tuple[Any, ...]) -> None:
        """Synchronize each device referenced by a collection of graphs."""
        devices: dict[str, Any] = {}
        for graph in graphs:
            devices[str(graph.device)] = graph.device
        for device in devices.values():
            wp.synchronize_device(device)
