# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph recording of the Warp frontend: :class:`GraphedCallable` records a call as CUDA graphs and replays it,
and :func:`eager` runs a call that cannot be recorded at its place in the recording.
"""

from __future__ import annotations

import contextlib
import functools
import gc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import warp as wp
import warp._src.context as _warp_context

from ..version import has_kit

if TYPE_CHECKING:
    from ...physics import PhysicsManager


def capturing(device: wp.Device | str | None = None) -> bool:
    """Whether a CUDA graph is being recorded on ``device`` (the current device by default).

    It reads Warp's table of the recordings in progress first, which costs ~20x less than ``Device.is_capturing``, so
    outside a recording the check spares the CUDA query; every recording in Isaac Lab starts through Warp.
    """
    runtime = _warp_context.runtime
    return runtime is not None and bool(runtime.captures) and wp.get_device(device).is_capturing


def is_recording() -> bool:
    """Whether a :class:`GraphedCallable` records, so that an :func:`eager` call repeats on each of its replays."""
    recording = GraphedCallable._recording
    return recording is not None and recording.capturing


def record_at_end(fn: Callable[[], None]) -> bool:
    """Record ``fn`` at the end of the :class:`GraphedCallable` being recorded, once however often it is requested.

    Returns:
        Whether a :class:`GraphedCallable` records; without one, the caller handles the replays itself.
    """
    recording = GraphedCallable._recording
    if recording is None:
        return False
    if fn not in recording.end_calls:
        recording.end_calls.append(fn)
    return True


def eager(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function that cannot be recorded from inside a :class:`GraphedCallable`.

    Outside a recording this is a plain call. While a :class:`GraphedCallable` records, the call ends the current
    graph, runs eagerly and starts the next graph, and every replay repeats it at the same place with the same
    arguments, so the arguments must stay valid across calls like the arrays a graph reads. Replays discard its result.

    Args:
        fn: The function to call.
        *args: Positional arguments of the call.
        **kwargs: Keyword arguments of the call.

    Returns:
        The result of the call.

    Raises:
        RuntimeError: If the current device is capturing a graph that no :class:`GraphedCallable` records.
    """
    recording = GraphedCallable._recording
    if recording is not None and recording.capturing:
        return recording.split(fn, args, kwargs)
    if capturing():
        name = getattr(fn, "__qualname__", type(fn).__name__)
        raise RuntimeError(
            f"'{name}' cannot be recorded, so it cannot run inside a graph capture that no GraphedCallable started."
        )
    return fn(*args, **kwargs)


def eager_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a method that cannot be recorded through :func:`eager`.

    A :class:`GraphedCallable` recording a call to the method replays it eagerly with the same arguments. An override
    in a subclass is decorated again.

    Args:
        method: The method to wrap.

    Returns:
        The wrapped method.
    """

    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        return eager(method, self, *args, **kwargs)

    return wrapper


class GraphedCallable:
    """A callable recorded as CUDA graphs on its first call with a given argument layout and replayed after.

    * The first call with a given argument layout records without an eager warm-up: the recording is its first
      execution. Later calls with the same array pointers and scalar values replay that recording and return the
      result of the recording call, so the callable returns buffers its owner keeps.
    * Calls that cannot be recorded go through :func:`eager`. A recording ends its graph at each of them, runs the
      call and starts the next graph; a replay launches the graphs and repeats the calls with their recorded
      arguments, in the recorded order.
    * :attr:`enabled` turns recording off process-wide, to run every part eagerly; it is on by default. A callable on
      a CPU device always runs eagerly.
    * :attr:`generation` advances when the terms change or the physics backend rebinds its buffers or stops (see
      :meth:`watch`), and every callable records again on its next call.
    * Called inside a graph capture that something else started, it runs inline, so its kernels become part of that
      capture.
    """

    enabled: ClassVar[bool] = True
    """Whether callables record CUDA graphs. Turn it off to run every recorded part eagerly."""

    generation: ClassVar[int] = 0
    """Counter of term changes, physics rebinds and stops; a callable recorded under an older value records again."""

    _callback_handles: ClassVar[list] = []

    _settle: ClassVar[Callable[[], None] | None] = None
    """The watched physics backend's forward, run before each recording (see :meth:`watch`)."""

    _recording: ClassVar[_Recording | None] = None
    """The recording in progress, which :func:`eager` calls split."""

    def __init__(self, fn: Callable[..., Any], device: wp.DeviceLike):
        """Initialize the callable.

        Args:
            fn: Callable to record. Its GPU work must be capturable, except for :func:`eager` calls.
            device: Warp device that runs the callable.
        """
        self._fn = fn
        self._device = wp.get_device(device)
        self._recordings: dict[tuple, tuple[list, Any]] = {}
        self._generation = GraphedCallable.generation

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the callable and return its result."""
        if capturing(self._device):
            return self._fn(*args, **kwargs)
        if not (GraphedCallable.enabled and self._device.is_cuda):
            return self._fn(*args, **kwargs)
        if self._generation != GraphedCallable.generation:
            self.reset()
            self._generation = GraphedCallable.generation
        key = _argument_key(args, kwargs)
        recorded = self._recordings.get(key)
        if recorded is None:
            self._recordings[key] = recorded = self._record(args, kwargs)
            return recorded[1]
        steps, result = recorded
        _replay(steps)
        return result

    def reset(self) -> None:
        """Drop the recordings; the callable records again on its next call."""
        if self._recordings:
            # graphs may still be executing; drain the device before releasing them
            wp.synchronize_device(self._device)
            self._recordings.clear()

    @classmethod
    def watch(cls, backend_manager: PhysicsManager) -> None:
        """Record every callable again after the physics backend rebinds its buffers or stops, and settle the backend
        before each recording: work it queued before (model changes, kinematics) runs then, once, so a recording holds
        only the work of its own calls.

        Args:
            backend_manager: Physics manager that dispatches the lifecycle events.
        """
        # imported here so that the utilities do not load the physics package
        from ...physics import PhysicsEvent

        cls.unwatch()
        cls._settle = backend_manager.forward
        cls._callback_handles = [
            backend_manager.register_callback(cls.invalidate, event)
            for event in (PhysicsEvent.PHYSICS_READY, PhysicsEvent.STOP)
        ]

    @classmethod
    def unwatch(cls) -> None:
        """Stop following the physics backend."""
        for handle in cls._callback_handles:
            handle.deregister()
        cls._callback_handles = []
        cls._settle = None

    @classmethod
    def invalidate(cls, payload: Any = None) -> None:
        """Record every callable again on its next call."""
        cls.generation += 1

    def _record(self, args: tuple, kwargs: dict[str, Any]) -> tuple[list, Any]:
        """Record the callable as its first run and return the recorded steps and the result."""
        with _record_calls(self._device) as recording:
            result = self._fn(*args, **kwargs)
        return recording.steps, result


@contextlib.contextmanager
def _record_calls(device: wp.Device):
    """Record the calls made inside as CUDA graphs split around :func:`eager` calls; the recording is their first run.

    Yields:
        The recording, whose steps replay the calls (see :func:`_replay`).
    """
    if GraphedCallable._settle is not None:
        # physics work queued before the recording runs now, once, instead of on every replay
        GraphedCallable._settle()
    recording = _Recording(device)
    enclosing, GraphedCallable._recording = GraphedCallable._recording, recording
    try:
        with _paused_gc(), wp.ScopedStream(recording.capture_stream):
            recording.begin()
            try:
                yield recording
                for fn in recording.end_calls:
                    fn()
            except BaseException:
                recording.abort()
                raise
            recording.end()
    finally:
        GraphedCallable._recording = enclosing


def _replay(steps: list) -> None:
    """Launch the graphs of a recording and repeat its eager calls, in the recorded order."""
    for step in steps:
        if isinstance(step, wp.Graph):
            wp.capture_launch(step)
        else:
            fn, args, kwargs = step
            fn(*args, **kwargs)


class _Recording:
    """The steps of one recording: CUDA graphs split around :func:`eager` calls, launched as they end."""

    def __init__(self, device: wp.Device):
        # graphs are captured on the capture stream and launched on the stream the callable runs on
        self.stream = wp.get_stream(device)
        self.capture_stream = self.stream
        self.mode = wp.CaptureMode.THREAD_LOCAL
        if has_kit():
            # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
            self.capture_stream = wp.stream_from_torch(torch.cuda.Stream(device=device.alias))
            self.mode = wp.CaptureMode.RELAXED
        self.steps: list = []
        # calls recorded after the recorded function (see record_at_end)
        self.end_calls: list = []
        self.capturing = False
        # identifies the recording to the buffers it refreshes, without keeping its graphs alive
        self.token = object()

    def begin(self) -> None:
        """Start the next graph."""
        wp.capture_begin(stream=self.capture_stream, capture_mode=self.mode)
        self.capturing = True

    def end(self) -> None:
        """End the current graph and launch it, since the recording is the first run of the callable."""
        self.capturing = False
        graph = wp.capture_end(stream=self.capture_stream)
        wp.capture_launch(graph, stream=self.stream)
        self.steps.append(graph)

    def abort(self) -> None:
        """End the current graph without launching it, after the callable raised."""
        if self.capturing:
            self.capturing = False
            with contextlib.suppress(Exception):
                wp.capture_end(stream=self.capture_stream)

    def split(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
        """Run a call that cannot be recorded between the current graph and the next one."""
        self.end()
        with wp.ScopedStream(self.stream, sync_enter=False):
            result = fn(*args, **kwargs)
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
    """Key a call by the pointer, shape, strides, dtype and device of its arrays and the values of its scalars."""
    return tuple(_value_key(value) for value in args) + tuple(
        (name, _value_key(value)) for name, value in kwargs.items()
    )


def _value_key(value: Any) -> Any:
    # views of one allocation share a pointer, so the layout tells them apart
    if isinstance(value, wp.array):
        return value.ptr, value.shape, value.strides, value.dtype, str(value.device)
    if isinstance(value, torch.Tensor):
        return value.data_ptr(), value.shape, value.stride(), value.dtype, value.device
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_value_key(item) for item in value)
    return id(value)
