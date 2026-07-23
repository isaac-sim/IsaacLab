# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp CUDA graph capture-or-replay utility."""

import os
from collections.abc import Callable
from typing import Any

import torch
import warp as wp

from isaaclab.utils.timer import Timer

SYNC_DEBUG_ENV_VAR = "ISAACLAB_SYNC_DEBUG"
"""Set ``ISAACLAB_SYNC_DEBUG=1`` to run eager and warm-up stage invocations under
``torch.cuda.set_sync_debug_mode("error")`` so hidden GPU->host syncs raise (CI/debug)."""

_SYNC_DEBUG = os.environ.get(SYNC_DEBUG_ENV_VAR, "0") == "1"


def _invoke_stage(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Run one stage callable, optionally under the sync-debug trap.

    The trap is the choke point that guards every current and future term inside a
    warp stage without per-function annotations; opt-outs suspend it explicitly
    (see :func:`isaaclab_experimental.utils.warp.any_env_set`).
    """
    if not _SYNC_DEBUG:
        return fn(*args, **kwargs)
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(2)
    try:
        return fn(*args, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode(prev)


CAPTURE_ENV_VAR = "ISAACLAB_WARP_CAPTURE"
"""Set ``ISAACLAB_WARP_CAPTURE=0`` to force every stage eager (debug / A-B validation)."""


class WarpGraphCache:
    """Execute Warp stages eagerly or through cached CUDA graphs.

    :meth:`call` uses the first invocation as an eager warm-up, captures the
    second invocation, and replays later invocations. This lets one-time
    initialization run outside capture without executing stateful manager work
    twice in one environment step. :meth:`capture_or_replay` retains the direct
    environment behavior of warming up and capturing on its first invocation.

    The return value from the capture run is cached and returned on every
    subsequent replay, ensuring captured stages return the same references
    (e.g. tensor views) as eager stages.

    Usage::

        cache = WarpGraphCache()
        result = cache.call("MyManager_step", my_warp_function)
        # uncaptured work here ...
        result2 = cache.call("OtherManager_step", my_other_function)
    """

    def __init__(self, *, enabled: bool = True, device: wp.DeviceLike = None):
        """Initialize the cache.

        Args:
            enabled: Whether eligible stages may use CUDA graph capture. The
                :data:`CAPTURE_ENV_VAR` environment variable can force this off.
            device: Warp device used by cached stages. CPU devices always run eagerly.
        """
        self._enabled = enabled and os.environ.get(CAPTURE_ENV_VAR, "1") != "0"
        self._device = wp.get_device(device) if device is not None else None
        self._graphs: dict[str, Any] = {}
        self._results: dict[str, Any] = {}
        self._capturable: dict[str, bool] = {}
        self._warmed: set[str] = set()

    @property
    def captured_stages(self) -> tuple[str, ...]:
        """Stages currently backed by a captured CUDA graph, sorted by name."""
        return tuple(sorted(self._graphs))

    @property
    def eager_groups(self) -> tuple[str, ...]:
        """Stage groups registered as non-capturable, sorted by name."""
        return tuple(sorted(group for group, capturable in self._capturable.items() if not capturable))

    def call(
        self,
        stage: str,
        fn: Callable[..., Any],
        /,
        *args: Any,
        output: Callable[[Any], Any] | None = None,
        timer: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run a Warp frontend stage eagerly or through its cached CUDA graph.

        The stage prefix before the first underscore identifies its capturability
        group. A group stays eager once any of its terms is registered as unsafe.
        Eligible stages run eagerly once, capture on their second invocation, and
        replay thereafter.

        Args:
            stage: Stage identifier in the form ``"GroupName_function_name"``.
            fn: Callable implementing the stage.
            *args: Positional arguments forwarded to :paramref:`fn`.
            output: Optional transform applied to the stage result after execution.
            timer: Whether to time the stage execution.
            **kwargs: Keyword arguments forwarded to :paramref:`fn`.

        Returns:
            The stage result, optionally transformed by :paramref:`output`.
        """
        group = stage.partition("_")[0]
        with Timer(name=stage, msg=f"{stage} took:", enable=timer, time_unit="us"):
            if not self._capture_enabled or not self.is_capturable(group):
                result = _invoke_stage(fn, args, kwargs)
            elif stage in self._graphs:
                wp.capture_launch(self._graphs[stage])
                result = self._results[stage]
            elif stage not in self._warmed:
                # The first real call doubles as warm-up. Stateful manager stages
                # must execute exactly once per environment step.
                result = _invoke_stage(fn, args, kwargs)
                self._warmed.add(stage)
            else:
                result = self._capture(stage, fn, args, kwargs)
                # CUDA graph capture records launches without executing them. Launch
                # the new graph so this logical manager call still advances once.
                wp.capture_launch(self._graphs[stage])
        return output(result) if output is not None else result

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
            The cached return value from the first (capture) invocation.
        """
        if kwargs is None:
            kwargs = {}
        if not self._capture_enabled:
            return _invoke_stage(fn, args, kwargs)
        graph = self._graphs.get(stage)
        if graph is not None:
            wp.capture_launch(graph)
            return self._results[stage]
        # Warm-up: run eagerly to flush first-call allocations / hasattr guards.
        _invoke_stage(fn, args, kwargs)
        return self._capture(stage, fn, args, kwargs)

    def _capture(
        self,
        stage: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Capture one already-warmed stage invocation."""
        with wp.ScopedCapture(device=self._device) as capture:
            result = fn(*args, **kwargs)
        self._graphs[stage] = capture.graph
        self._results[stage] = result
        self._warmed.add(stage)
        # One grep-able line per stage so runs can assert capture coverage.
        print(f"[INFO] WarpGraphCache: captured stage '{stage}'")
        return result

    def register_capturability(self, key: str, capturable: bool) -> None:
        """Register conservative capture eligibility for a stage group.

        Args:
            key: Stage group identifier.
            capturable: Whether every operation registered under the group is safe
                during CUDA graph capture.
        """
        self._capturable[key] = self._capturable.get(key, True) and capturable

    def is_capturable(self, key: str) -> bool:
        """Return whether a stage group is eligible for capture."""
        return self._capturable.get(key, True)

    @property
    def _capture_enabled(self) -> bool:
        """Return whether this cache can capture on its configured device."""
        device = self._device if self._device is not None else wp.get_device()
        return self._enabled and device.is_cuda

    def invalidate(self, stage: str | None = None) -> None:
        """Drop cached graph(s). If *stage* is ``None``, drop all."""
        if stage is None:
            self._graphs.clear()
            self._results.clear()
            self._warmed.clear()
        else:
            self._graphs.pop(stage, None)
            self._results.pop(stage, None)
            self._warmed.discard(stage)
