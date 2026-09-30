# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph capture and replay for Warp frontend stages."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
import warp as wp
from isaaclab_newton.physics import NewtonQueries

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit

CAPTURE_ENV_VAR = "ISAACLAB_WARP_CAPTURE"
"""Set ``ISAACLAB_WARP_CAPTURE=0`` to run every stage eagerly (debugging and eager-vs-captured comparisons)."""

SYNC_DEBUG_ENV_VAR = "ISAACLAB_SYNC_DEBUG"
"""Set ``ISAACLAB_SYNC_DEBUG=1`` to run eager stage calls under ``torch.cuda.set_sync_debug_mode("error")``,
so a hidden GPU-to-host synchronization inside a stage raises instead of stalling silently."""


class WarpGraphCache:
    """Run Warp frontend stages eagerly or through cached CUDA graphs.

    Graphs follow the physics backend's capture lifecycle:

    * A stage is recorded without an eager warm-up: its first execution is the launch of the
      newly recorded graph, so a stateful stage advances exactly once per call. Stage owners
      allocate their buffers before the first call.
    * Recording waits until :meth:`arm`. Environments arm the cache on each step, so stages
      called only while constructing or resetting the environment run eagerly until then.
    * A graph is keyed by the pointers and scalar values of the stage arguments. Passing a
      reallocated array, or a different scalar, records the stage again.
    * :meth:`invalidate` drops every graph and holds recording until the next :meth:`arm`.
      :meth:`invalidate_on` does so whenever the physics backend rebinds its buffers or stops.

    A stage group is the stage name up to its first underscore. A group registered as not
    capturable, a CPU device, or ``ISAACLAB_WARP_CAPTURE=0`` runs its stages eagerly.
    """

    def __init__(self, device: wp.DeviceLike = None):
        """Initialize the cache.

        Args:
            device: Warp device that runs the stages. Defaults to the current Warp device.
        """
        self._device = wp.get_device(device)
        self._enabled = self._device.is_cuda and os.environ.get(CAPTURE_ENV_VAR, "1") != "0"
        self._sync_debug = os.environ.get(SYNC_DEBUG_ENV_VAR, "0") == "1"
        self._armed = False
        self._graphs: dict[str, tuple[tuple, wp.Graph, Any]] = {}
        self._capturable: dict[str, bool] = {}
        self._callback_handles = []

    @property
    def captured_stages(self) -> tuple[str, ...]:
        """Stages currently backed by a recorded graph, sorted by name."""
        return tuple(sorted(self._graphs))

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
        """Run a stage eagerly or through its recorded graph.

        Args:
            stage: Stage identifier in the form ``"GroupName_function_name"``.
            fn: Callable implementing the stage. Its GPU work must be capturable when the group is.
            *args: Positional arguments forwarded to :paramref:`fn`.
            output: Transform applied to the stage result on every call, outside the graph.
            timer: Whether to time the stage.
            **kwargs: Keyword arguments forwarded to :paramref:`fn`.

        Returns:
            The stage result, optionally transformed by :paramref:`output`. A replayed stage returns
            the result of the call that recorded it.
        """
        with Timer(name=stage, msg=f"{stage} took:", enable=timer, time_unit="us"):
            if not self._enabled or not self.is_capturable(stage.partition("_")[0]):
                result = self._run_eager(fn, args, kwargs)
            else:
                key = _argument_key(args, kwargs)
                recorded = self._graphs.get(stage)
                if recorded is not None and recorded[0] == key:
                    wp.capture_launch(recorded[1])
                    result = recorded[2]
                elif self._armed:
                    graph, result = self._record(fn, args, kwargs)
                    self._graphs[stage] = (key, graph, result)
                    wp.capture_launch(graph)
                else:
                    result = self._run_eager(fn, args, kwargs)
        return output(result) if output is not None else result

    def arm(self) -> None:
        """Allow capturable stages to record on their next call."""
        self._armed = True

    def invalidate(self, group: str | None = None) -> None:
        """Drop recorded graphs.

        Args:
            group: Stage group whose graphs are dropped; they record again on their next armed call.
                Defaults to None, which drops every graph and holds recording until the next :meth:`arm`.
        """
        stages = [stage for stage in self._graphs if group is None or stage.partition("_")[0] == group]
        if not stages:
            if group is None:
                self._armed = False
            return
        # Graphs may still be executing; drain the device before releasing them.
        wp.synchronize_device(self._device)
        for stage in stages:
            del self._graphs[stage]
        if group is None:
            self._armed = False

    def invalidate_on(self, physics_manager: type[PhysicsManager]) -> None:
        """Invalidate every graph when the physics backend rebinds its buffers or stops.

        Args:
            physics_manager: Physics manager class that dispatches the lifecycle events.
        """
        for event in (PhysicsEvent.PHYSICS_READY, PhysicsEvent.STOP):
            self._callback_handles.append(physics_manager.register_callback(self._on_physics_event, event))

    def close(self) -> None:
        """Deregister the physics callbacks and drop every graph."""
        for handle in self._callback_handles:
            handle.deregister()
        self._callback_handles.clear()
        self.invalidate()

    def register_capturability(self, group: str, capturable: bool) -> None:
        """Register whether a stage group is safe to record.

        One unsafe registration keeps the group eager.

        Args:
            group: Stage group identifier.
            capturable: Whether the registered work is safe during CUDA graph capture.
        """
        self._capturable[group] = self._capturable.get(group, True) and capturable

    def is_capturable(self, group: str) -> bool:
        """Return whether a stage group is eligible for recording."""
        return self._capturable.get(group, True)

    def _on_physics_event(self, payload: Any) -> None:
        self.invalidate()

    def _record(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> tuple[wp.Graph, Any]:
        """Record one stage call without executing it."""
        result = None

        def capture_target() -> None:
            nonlocal result
            result = fn(*args, **kwargs)

        graph = NewtonQueries.capture_graph(str(self._device), capture_target, relaxed=has_kit())
        return graph, result

    def _run_eager(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
        """Run a stage eagerly, under the synchronization trap when requested."""
        if not self._sync_debug:
            return fn(*args, **kwargs)
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)


def _argument_key(args: tuple, kwargs: dict[str, Any]) -> tuple:
    """Key a stage call by the pointers of its arrays and the values of its scalars."""
    return tuple(_value_key(value) for value in args) + tuple(
        (name, _value_key(value)) for name, value in kwargs.items()
    )


def _value_key(value: Any) -> Any:
    if isinstance(value, wp.array):
        return value.ptr
    if isinstance(value, torch.Tensor):
        return value.data_ptr()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return id(value)
