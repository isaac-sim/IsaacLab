# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph capture of Warp frontend stages: a :func:`captured` method records its calls and replays them, and
:func:`eager` runs a call that cannot be recorded at its place in the recording.

.. code-block:: python

    @captured
    def compute(self, dt):
        eager(term_cfg.func, self._env, term_cfg.out, **term_cfg.params)
"""

from __future__ import annotations

import contextlib
import functools
import gc
import os
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import warp as wp

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.utils.version import has_kit

SYNC_DEBUG_ENV_VAR = "ISAACLAB_SYNC_DEBUG"
"""Set ``ISAACLAB_SYNC_DEBUG=1`` to run eager stage calls under ``torch.cuda.set_sync_debug_mode("error")``,
so a hidden GPU-to-host synchronization inside a stage raises instead of stalling silently."""


def captured(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a method through a :class:`CapturedStage` stored on its instance.

    The method body is the plain eager code of the stage, with the calls that cannot be recorded wrapped in
    :func:`eager`. Its instance must expose ``device``. The instance keeps its stages in ``_captured_stages``,
    keyed by the decorated method, so an override and the method it extends own separate stages.

    Args:
        method: The method to decorate.
    """

    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        stages = self.__dict__.setdefault("_captured_stages", {})
        stage = stages.get(wrapper)
        if stage is None:
            stage = stages[wrapper] = CapturedStage(method, self.device)
        return stage(self, *args, **kwargs)

    return wrapper


def eager(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function that cannot be recorded from inside a :class:`CapturedStage`.

    Outside a recording this is a plain call. While a stage records, the call ends the current graph, runs eagerly
    and starts the next graph, and every replay of the stage repeats it at the same place with the same arguments,
    so the arguments must stay valid across calls like the arrays a graph reads. Replays discard its result.

    Args:
        fn: The function to call.
        *args: Positional arguments of the call.
        **kwargs: Keyword arguments of the call.

    Returns:
        The result of the call.

    Raises:
        RuntimeError: If the current device is capturing a graph that no :class:`CapturedStage` records.
    """
    recording = CapturedStage._recording
    if recording is not None and recording.capturing:
        return recording.split(fn, args, kwargs)
    if wp.get_device().is_capturing:
        name = getattr(fn, "__qualname__", type(fn).__name__)
        raise RuntimeError(
            f"'{name}' cannot be recorded, so it cannot run inside a graph capture that no CapturedStage started."
        )
    return fn(*args, **kwargs)


def reset_captured_stages(owner: Any) -> None:
    """Drop the graphs recorded by the :func:`captured` methods of an instance."""
    for stage in owner.__dict__.get("_captured_stages", {}).values():
        stage.reset()


class CapturedStage:
    """A stage of the Warp frontend, run eagerly or through CUDA graphs recorded from its own calls.

    * The first call with a given argument layout records the stage without an eager warm-up: the recording is its
      first execution. Later calls with the same array pointers and scalar values replay that recording and return
      the result of the recording call, so a stage returns buffers its owner keeps.
    * Calls that cannot be recorded go through :func:`eager`. A recording ends its graph at each of them, runs the
      call and starts the next graph; a replay launches the graphs and repeats the calls with their recorded
      arguments, in the recorded order.
    * :attr:`enabled` is process-wide and set by the Warp environments. A stage runs eagerly while it is False and
      on CPU devices.
    * :attr:`generation` advances when the physics backend rebinds its buffers or stops (see :meth:`invalidate_on`),
      and every stage records again on its next call.
    * A stage called while another stage records runs inline, so its graphs and eager calls become part of that
      recording.
    """

    enabled: ClassVar[bool] = False
    """Whether stages record CUDA graphs. The Warp environments set it; the torch frontend never captures."""

    generation: ClassVar[int] = 0
    """Counter of physics rebinds and stops; a stage recorded under an older value records again."""

    _callback_handles: ClassVar[list] = []

    _recording: ClassVar[_Recording | None] = None
    """The recording in progress, which :func:`eager` calls split."""

    def __init__(self, fn: Callable[..., Any], device: wp.DeviceLike):
        """Initialize the stage.

        Args:
            fn: Callable implementing the stage. Its GPU work must be capturable, except for :func:`eager` calls.
            device: Warp device that runs the stage.
        """
        self._fn = fn
        self._device = wp.get_device(device)
        self._sync_debug = os.environ.get(SYNC_DEBUG_ENV_VAR, "0") == "1"
        self._recordings: dict[tuple, tuple[list, Any]] = {}
        self._generation = CapturedStage.generation

    @property
    def num_graphs(self) -> int:
        """Number of graphs currently recorded for this stage."""
        return sum(isinstance(step, wp.Graph) for steps, _ in self._recordings.values() for step in steps)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the stage and return its result."""
        if self._device.is_capturing:
            return self._fn(*args, **kwargs)
        if not (CapturedStage.enabled and self._device.is_cuda):
            return self._run_eager(self._fn, args, kwargs)
        if self._generation != CapturedStage.generation:
            self.reset()
            self._generation = CapturedStage.generation
        key = _argument_key(args, kwargs)
        recorded = self._recordings.get(key)
        if recorded is None:
            self._recordings[key] = recorded = self._record(args, kwargs)
            return recorded[1]
        steps, result = recorded
        for step in steps:
            if isinstance(step, wp.Graph):
                wp.capture_launch(step)
            else:
                self._run_eager(*step)
        return result

    def reset(self) -> None:
        """Drop the recordings; the stage records again on its next call."""
        if self._recordings:
            # graphs may still be executing; drain the device before releasing them
            wp.synchronize_device(self._device)
            self._recordings.clear()

    @classmethod
    def invalidate_on(cls, physics_manager: type[PhysicsManager]) -> None:
        """Record every stage again after the physics backend rebinds its buffers or stops.

        Args:
            physics_manager: Physics manager class that dispatches the lifecycle events.
        """
        cls._callback_handles = [
            physics_manager.register_callback(cls.invalidate, event)
            for event in (PhysicsEvent.PHYSICS_READY, PhysicsEvent.STOP)
        ]

    @classmethod
    def disable(cls) -> None:
        """Run every stage eagerly again and stop following the physics backend."""
        for handle in cls._callback_handles:
            handle.deregister()
        cls._callback_handles = []
        cls.enabled = False

    @classmethod
    def invalidate(cls, payload: Any = None) -> None:
        """Record every stage again on its next call."""
        cls.generation += 1

    def _record(self, args: tuple, kwargs: dict[str, Any]) -> tuple[list, Any]:
        """Record the stage as its first run and return the recorded steps and the result."""
        recording = _Recording(self._device, self._run_eager)
        enclosing, CapturedStage._recording = CapturedStage._recording, recording
        try:
            with _paused_gc(), wp.ScopedStream(recording.capture_stream):
                recording.begin()
                try:
                    result = self._fn(*args, **kwargs)
                except BaseException:
                    recording.abort()
                    raise
                recording.end()
        finally:
            CapturedStage._recording = enclosing
        return recording.steps, result

    def _run_eager(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
        """Run eagerly, under the synchronization trap when requested."""
        if not self._sync_debug:
            return fn(*args, **kwargs)
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)


class _Recording:
    """The steps of one stage recording: CUDA graphs split around :func:`eager` calls, launched as they end."""

    def __init__(self, device: wp.Device, run_eager: Callable[[Callable[..., Any], tuple, dict], Any]):
        # graphs are captured on the capture stream and launched on the stream the stage runs on
        self.stream = wp.get_stream(device)
        self.capture_stream = self.stream
        self.mode = wp.CaptureMode.THREAD_LOCAL
        if has_kit():
            # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
            self.capture_stream = wp.stream_from_torch(torch.cuda.Stream(device=device.alias))
            self.mode = wp.CaptureMode.RELAXED
        self.steps: list = []
        self.capturing = False
        self._run_eager = run_eager

    def begin(self) -> None:
        """Start the next graph."""
        wp.capture_begin(stream=self.capture_stream, capture_mode=self.mode)
        self.capturing = True

    def end(self) -> None:
        """End the current graph and launch it, since the recording is the first run of the stage."""
        self.capturing = False
        graph = wp.capture_end(stream=self.capture_stream)
        wp.capture_launch(graph, stream=self.stream)
        self.steps.append(graph)

    def abort(self) -> None:
        """End the current graph without launching it, after the stage raised."""
        if self.capturing:
            self.capturing = False
            with contextlib.suppress(Exception):
                wp.capture_end(stream=self.capture_stream)

    def split(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
        """Run a call that cannot be recorded between the current graph and the next one."""
        self.end()
        with wp.ScopedStream(self.stream, sync_enter=False):
            result = self._run_eager(fn, args, kwargs)
        self.steps.append((fn, args, kwargs))
        self.begin()
        return result


@contextlib.contextmanager
def _paused_gc():
    """Keep collection-driven frees out of a recording, as ``NewtonQueries.capture_graph`` does.

    Objects allocated during the recording stay in the youngest generation, so collecting that generation afterwards
    frees the recording's garbage without the full collection of the whole heap.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
            gc.collect(0)


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
