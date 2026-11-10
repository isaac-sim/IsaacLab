# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph recording of the Warp frontend: an environment calls the steps of its step through
:class:`GraphedCalls`, which records each call as CUDA graphs (see :class:`GraphedCallable`) and replays it, and
records the consecutive calls of a block as one graph. :func:`eager` runs a call that cannot be recorded at its place
in the recording.

.. code-block:: python

    # on the torch frontend the calls are plain calls and the block does nothing
    self._capture = GraphedCalls(self.device, self.frontend.captures, self.sim.backend_manager)
    self._capture(self.action_manager.apply_action)  # the first call records, later calls replay
    with self._capture.block("post_physics"):  # the calls inside replay as one graph
        self.reset_buf = self._capture(self.termination_manager.compute)
        self.reward_buf = self._capture(self.reward_manager.compute, dt=self.step_dt)
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
import warp._src.context as _warp_context

from ...physics import PhysicsEvent, PhysicsManager
from ..version import has_kit

SYNC_DEBUG_ENV_VAR = "ISAACLAB_SYNC_DEBUG"
"""Set ``ISAACLAB_SYNC_DEBUG=1`` to run eager calls under ``torch.cuda.set_sync_debug_mode("error")``, so a hidden
GPU-to-host synchronization inside a recorded part raises instead of stalling silently."""


def recordings() -> dict | None:
    """Warp's table of the CUDA graph recordings in progress, keyed by capture id, or None before Warp initializes.

    The table is empty while nothing records. Reading it costs ~20x less than ``Device.is_capturing``, which also asks
    CUDA about recordings Warp did not start; every recording in Isaac Lab starts through Warp.
    """
    runtime = _warp_context.runtime
    return None if runtime is None else runtime.captures


def capturing(device: wp.Device | str | None = None) -> bool:
    """Whether a CUDA graph is being recorded on ``device`` (the current device by default).

    It reads :func:`recordings` first, so outside a recording the check spares the CUDA query of
    ``Device.is_capturing``.
    """
    return bool(recordings()) and wp.get_device(device).is_capturing


def recording_key(captures: dict) -> object:
    """The identity of the recording in progress, given a non-empty :func:`recordings` table.

    A :class:`GraphedCallable` recording spans the graphs it splits around eager calls, so it is the identity while one
    records; any other recording is identified by its captures.
    """
    recording = GraphedCallable._recording
    return tuple(captures) if recording is None else recording.token


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
    if GraphedCallable._replaying:
        # a block replaying its recording runs the call from the recording when it ends
        return None
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

    :class:`GraphedCalls` keeps one per call an environment makes in its step, so the graph boundaries of a step are
    in the environment; the managers and terms it calls are plain code.

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

    _replaying: ClassVar[bool] = False
    """Whether a :class:`GraphedCalls` block replays, so that the calls made inside it run when it ends."""

    def __init__(self, fn: Callable[..., Any], device: wp.DeviceLike):
        """Initialize the callable.

        Args:
            fn: Callable to record. Its GPU work must be capturable, except for :func:`eager` calls.
            device: Warp device that runs the callable.
        """
        self._fn = fn
        self._device = wp.get_device(device)
        self._sync_debug = os.environ.get(SYNC_DEBUG_ENV_VAR, "0") == "1"
        self._recordings: dict[tuple, tuple[list, Any]] = {}
        self._generation = GraphedCallable.generation

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the callable and return its result."""
        if capturing(self._device):
            return self._fn(*args, **kwargs)
        if not (GraphedCallable.enabled and self._device.is_cuda):
            return self._run_eager(self._fn, args, kwargs)
        if self._generation != GraphedCallable.generation:
            self.reset()
            self._generation = GraphedCallable.generation
        key = _argument_key(args, kwargs)
        recorded = self._recordings.get(key)
        if recorded is None:
            self._recordings[key] = recorded = self._record(args, kwargs)
            return recorded[1]
        steps, result = recorded
        _replay(steps, self._run_eager)
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
        with _record_calls(self._device, self._run_eager) as recording:
            result = self._fn(*args, **kwargs)
        return recording.steps, result

    def _run_eager(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
        """Run eagerly, under the synchronization trap when requested."""
        return _run_eager(fn, args, kwargs, self._sync_debug)


class GraphedCalls:
    """The calls an environment makes in its step, recorded as CUDA graphs when recording is on.

    ``calls(fn, *args, **kwargs)`` returns ``fn(*args, **kwargs)``. With recording on (the Warp frontend on a CUDA
    device), the first call of ``fn`` with given scalar arguments records it (see :class:`GraphedCallable`) and later
    calls replay it. An array argument must be the array the recording read; an array the caller hands anew on each
    call, such as the actions of a step, is passed as :meth:`staged` and copied into a buffer the recording reads.

    Inside ``with calls.block(name):`` the calls are recorded together, so a replay launches one graph for them, split
    only where a call cannot be recorded (see :func:`eager`). A replay runs the block when the block ends: inside it,
    each call returns its result from the recording, so the code of a block does its GPU work only through the calls
    and does not read their results on the host. A block called with other arrays than recorded runs its calls
    eagerly and records again on its next run.
    """

    def __init__(self, device: wp.DeviceLike, enabled: bool, backend_manager: PhysicsManager | None = None):
        """Initialize the calls.

        Args:
            device: Device the calls run on.
            enabled: Whether to record; without it every call is a plain call.
            backend_manager: Physics manager whose rebinds and stops make the calls record again (see
                :meth:`GraphedCallable.watch`).
        """
        self._device = wp.get_device(device)
        self._enabled = enabled and self._device.is_cuda
        self._sync_debug = os.environ.get(SYNC_DEBUG_ENV_VAR, "0") == "1"
        self._graphed: dict = {}
        """The recorded calls: callable -> (its :class:`GraphedCallable`, its input buffers by argument)."""
        self._blocks: dict[str, _Block] = {}
        self._block: _Block | None = None
        """The block running."""
        self._watching = self._enabled and backend_manager is not None
        if self._watching:
            GraphedCallable.watch(backend_manager)

    def __call__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``fn``: recorded on its first call and replayed after when recording is on.

        Args:
            fn: The function to call.
            *args: Positional arguments of the call.
            **kwargs: Keyword arguments of the call.

        Returns:
            The result of the call.
        """
        if not self._enabled:
            return fn(*args, **kwargs)
        if self._block is not None:
            return self._block.call(fn, args, kwargs)
        entry = self._graphed.get(fn)
        if entry is None:
            entry = self._graphed[fn] = (GraphedCallable(fn, self._device), {})
        graphed, inputs = entry
        args = tuple(self._input(fn, inputs, index, value) for index, value in enumerate(args))
        kwargs = {name: self._input(fn, inputs, name, value) for name, value in kwargs.items()}
        return graphed(*args, **kwargs)

    def block(self, name: str) -> contextlib.AbstractContextManager:
        """The calls made inside, recorded as one graph; a block inside a block is part of it.

        Args:
            name: Name of the block, which keys its recording.

        Returns:
            The context of the block.
        """
        if not self._enabled or self._block is not None:
            return _NO_BLOCK
        block = self._blocks.get(name)
        if block is None:
            block = self._blocks[name] = _Block(self)
        return block

    def staged(self, array: torch.Tensor | wp.array) -> Any:
        """Mark an array argument the caller hands anew on each call, such as the actions of a step.

        The call copies it into a buffer of its own, which the recording reads. Without recording it is the array.
        """
        return _Staged(array) if self._enabled else array

    def close(self):
        """Drop the recordings and stop following the physics backend."""
        if self._watching:
            GraphedCallable.unwatch()
            self._watching = False
        if self._graphed or self._blocks:
            # graphs may still be executing; drain the device before releasing them
            wp.synchronize_device(self._device)
        self._graphed.clear()
        self._blocks.clear()

    def _input(self, fn: Callable[..., Any], inputs: dict, slot: int | str, value: Any) -> Any:
        """The array the recording reads for an argument: a staged array is copied into the call's buffer for it, any
        other array must be the one recorded.

        Raises:
            ValueError: If an array argument that is not staged is another array than the recording read.
        """
        if isinstance(value, _Staged):
            buffer = inputs.get(slot)
            if buffer is None:
                array = value.array
                buffer = inputs[slot] = array.clone() if isinstance(array, torch.Tensor) else wp.clone(array)
            elif isinstance(buffer, torch.Tensor):
                buffer.copy_(value.array)
            else:
                wp.copy(buffer, value.array)
            return buffer
        if not isinstance(value, (torch.Tensor, wp.array)):
            return value
        if inputs.setdefault(slot, _value_key(value)) != _value_key(value):
            raise ValueError(
                f"{getattr(fn, '__qualname__', fn)}: argument {slot!r} is another array than the recording read. Pass"
                " the same array on each call, or stage it with GraphedCalls.staged()."
            )
        return value


class _Staged:
    """An array argument a :class:`GraphedCalls` call copies into a buffer the recording reads (see
    :meth:`GraphedCalls.staged`)."""

    __slots__ = ("array",)

    def __init__(self, array: torch.Tensor | wp.array):
        self.array = array


class _Block:
    """A :meth:`GraphedCalls.block`: its recording and the calls it made, replayed when the block ends."""

    def __init__(self, calls: GraphedCalls):
        self._calls = calls
        self._steps: list | None = None
        self._generation = GraphedCallable.generation
        self._made: list = []
        """The calls of the recording, as (callable, argument key, result), in order."""
        self._recording = None
        """The context of the recording in progress, and the recording."""
        self._replayed: list | None = None
        """The calls of the replay in progress, as (callable, args, kwargs), in order."""
        self._depth = 0

    def __enter__(self) -> _Block:
        if not GraphedCallable.enabled:
            return self
        calls = self._calls
        calls._block = self
        if self._steps is not None and self._generation == GraphedCallable.generation:
            self._replayed = []
            GraphedCallable._replaying = True
            return self
        if self._steps is not None:
            wp.synchronize_device(calls._device)
        self._steps, self._made, self._generation = None, [], GraphedCallable.generation
        context = _record_calls(calls._device, functools.partial(_run_eager, sync_debug=calls._sync_debug))
        self._recording = (context, context.__enter__())
        return self

    def call(self, fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
        """Make a call of the block: recorded inline, or, in a replay, answered from the recording."""
        if self._replayed is not None:
            index = len(self._replayed)
            self._replayed.append((fn, args, kwargs))
            return self._made[index][2] if index < len(self._made) else None
        if self._depth:
            # a call that a call of the block makes is part of that call
            return fn(*args, **kwargs)
        self._depth += 1
        try:
            result = fn(*args, **kwargs)
        finally:
            self._depth -= 1
        self._made.append((fn, _argument_key(args, kwargs), result))
        return result

    def __exit__(self, exc_type, exc, traceback) -> bool:
        calls = self._calls
        calls._block = None
        if self._recording is not None:
            (context, recording), self._recording = self._recording, None
            context.__exit__(exc_type, exc, traceback)
            if exc_type is None:
                self._steps = recording.steps
            return False
        if self._replayed is None:
            return False
        replayed, self._replayed = self._replayed, None
        GraphedCallable._replaying = False
        if exc_type is not None:
            return False
        made = [(fn, key) for fn, key, _ in self._made]
        if made == [(fn, _argument_key(args, kwargs)) for fn, args, kwargs in replayed]:
            _replay(self._steps, functools.partial(_run_eager, sync_debug=calls._sync_debug))
            return False
        self._steps = None
        if [fn for fn, _ in made] != [fn for fn, _, _ in replayed]:
            raise RuntimeError("A block of recorded calls made other calls than its recording.")
        # the same calls on other arrays: run them now, and record the block again on its next run
        for fn, args, kwargs in replayed:
            fn(*args, **kwargs)
        return False


_NO_BLOCK = contextlib.nullcontext()
"""The block of calls that are not recorded, or that are part of an enclosing block."""


@contextlib.contextmanager
def _record_calls(device: wp.Device, run_eager: Callable[[Callable[..., Any], tuple, dict], Any]):
    """Record the calls made inside as CUDA graphs split around :func:`eager` calls; the recording is their first run.

    Yields:
        The recording, whose steps replay the calls (see :func:`_replay`).
    """
    if GraphedCallable._settle is not None:
        # physics work queued before the recording runs now, once, instead of on every replay
        GraphedCallable._settle()
    recording = _Recording(device, run_eager)
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


def _replay(steps: list, run_eager: Callable[[Callable[..., Any], tuple, dict], Any]) -> None:
    """Launch the graphs of a recording and repeat its eager calls, in the recorded order."""
    for step in steps:
        if isinstance(step, wp.Graph):
            wp.capture_launch(step)
        else:
            run_eager(*step)


def _run_eager(fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any], sync_debug: bool = False) -> Any:
    """Run eagerly, under the synchronization trap when requested."""
    if not sync_debug:
        return fn(*args, **kwargs)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        return fn(*args, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


class _Recording:
    """The steps of one recording: CUDA graphs split around :func:`eager` calls, launched as they end."""

    def __init__(
        self,
        device: wp.Device,
        run_eager: Callable[[Callable[..., Any], tuple, dict], Any],
    ):
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
        self._run_eager = run_eager

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
    """Key a call by the pointers of its arrays and the values of its scalars."""
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
    if isinstance(value, (list, tuple)):
        return tuple(_value_key(item) for item in value)
    return id(value)
