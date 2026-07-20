# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp CUDA graph capture-or-replay utility."""

from collections.abc import Callable
from typing import Any

import warp as wp


class WarpGraphCache:
    """Execute Warp stages eagerly or through cached CUDA graphs.

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
        result = cache.call("my_stage", my_warp_function)
        # uncaptured work here ...
        result2 = cache.call("my_stage_post", my_other_function)
    """

    def __init__(self, *, enabled: bool = True):
        """Initialize the execution cache.

        Args:
            enabled: Whether to capture stages. When false, stages execute eagerly.
        """
        self._enabled = enabled
        self._graphs: dict[str, Any] = {}
        self._results: dict[str, Any] = {}
        self._capturable: dict[str, bool] = {}

    def call(
        self,
        stage: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        capture: bool | None = None,
        group: str | None = None,
    ) -> Any:
        """Execute a stage using its registered capture policy.

        Args:
            stage: Unique stage identifier.
            fn: Callable implementing the stage.
            args: Positional arguments forwarded to :paramref:`fn`.
            kwargs: Keyword arguments forwarded to :paramref:`fn`.
            capture: Whether capture is requested for this call. When omitted,
                follows the cache-wide :attr:`_enabled` setting.
            group: Optional capturability group. This lets an owner register one
                decision for several stages, such as all calls on one manager.

        Returns:
            The eager stage result or cached capture result.
        """
        if kwargs is None:
            kwargs = {}
        capture_requested = self._enabled if capture is None else capture
        capture_key = stage if group is None else group
        if not self._enabled or not capture_requested or not self.is_capturable(capture_key):
            return fn(*args, **kwargs)
        return self._capture_or_replay(stage, fn, args, kwargs)

    def capture_or_replay(
        self,
        stage: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
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
            The eager result when disabled, otherwise the cached result from capture.
        """
        return self.call(stage, fn, args, kwargs, capture=True)

    def register_capturability(self, key: str, capturable: bool) -> None:
        """Register capture eligibility for a stage or owner group.

        Repeated registrations are conservative: once a key is non-capturable,
        later registrations cannot silently re-enable it.

        Args:
            key: Stage or group identifier.
            capturable: Whether every operation registered under the key is safe
                during CUDA graph capture.
        """
        self._capturable[key] = self._capturable.get(key, True) and capturable

    def is_capturable(self, key: str) -> bool:
        """Return whether a stage or group is eligible for capture."""
        return self._capturable.get(key, True)

    def invalidate(self, stage: str | None = None) -> None:
        """Drop cached graph(s). If *stage* is ``None``, drop all."""
        if stage is None:
            self._graphs.clear()
            self._results.clear()
        else:
            self._graphs.pop(stage, None)
            self._results.pop(stage, None)

    def _capture_or_replay(
        self,
        stage: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Capture a stage on first use and replay it afterward."""
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
