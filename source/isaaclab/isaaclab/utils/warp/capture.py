# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph recording of the Warp frontend: :class:`GraphedCallable` records a call as a CUDA graph and replays it.
A method that cannot be recorded is marked with :func:`eager_method`; a recorded call that calls one runs plainly
instead.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.utils._pytree as _pytree
import warp as wp
import warp._src.context as _warp_context
from torch.utils._python_dispatch import TorchDispatchMode

from ..version import has_kit

logger = logging.getLogger(__name__)


def capturing(device: wp.Device | str | None = None) -> bool:
    """Whether a graph is being recorded on ``device`` (the current device by default), a CUDA graph or Warp's CPU one.

    It reads Warp's records of the captures in progress first, which costs ~20x less than ``Device.is_capturing``, so
    outside a recording the check spares the CUDA query; every recording in Isaac Lab starts through Warp. Warp's CPU
    capture is not in its CUDA capture table and is read from the runtime's private ``_apic_capture``.
    """
    if not capture_in_progress():
        return False
    return wp.get_device(device).is_capturing


def capture_in_progress() -> bool:
    """Whether Warp records a graph on any device, a CUDA graph or its CPU one."""
    runtime = _warp_context.runtime
    return runtime is not None and bool(runtime.captures or runtime._apic_capture is not None)


class _EagerCall(BaseException):
    """Raised by an :func:`eager_method` method called inside a recording, which the :class:`GraphedCallable`
    abandons.

    It derives from :class:`BaseException`, so the ``except Exception`` handlers of the recorded code let it through.
    """

    def __init__(self, fn: Callable[..., Any]):
        super().__init__(_name(fn))


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__qualname__", type(fn).__name__)


def eager_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method that cannot be recorded, such as one that reads device data on the host or runs torch work.

    Outside a recording the method runs as a plain call. A :class:`GraphedCallable` that calls it while recording
    abandons the recording, which launches nothing, and runs plainly from then on, this call included. An override in a
    subclass is decorated again.

    Args:
        method: The method to mark.

    Returns:
        The wrapped method.

    Raises:
        RuntimeError: If the method is called while the current device captures a graph that no
            :class:`GraphedCallable` records.
    """

    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        recording = GraphedCallable._recording
        if recording is not None and recording.capturing:
            raise _EagerCall(method)
        if capturing():
            raise RuntimeError(
                f"'{_name(method)}' cannot be recorded, so it cannot run inside a graph capture that no"
                " GraphedCallable started."
            )
        return method(self, *args, **kwargs)

    return wrapper


class GraphedCallable:
    """A callable recorded as a CUDA graph on its first call with a given argument layout and replayed after.

    * The first call with a given argument layout records without an eager warm-up: the recording is its first
      execution. Later calls with the same array pointers and scalar values replay that recording and return the
      result of the recording call, so the callable returns buffers its owner keeps.
    * Methods that cannot be recorded are marked with :func:`eager_method`. A recording that calls one is abandoned
      without launching anything, and the callable runs plainly from then on, starting with this call; it records
      again after :meth:`reset`. Known limitation: the host work the callable did before that call, such as a Python
      counter it advanced, ran once in the abandoned recording and runs again in the plain call, on that first call
      only; its device work ran only in the plain call.
    * A callable on a CPU device always runs eagerly.
    * Its owner calls :meth:`reset` when what a recording holds changes, such as the physics buffers after a rebind,
      and the callable records again on its next call.
    * Called inside a graph capture that something else started, it runs inline, so its kernels become part of that
      capture.
    """

    _recording: ClassVar[_Recording | None] = None
    """The recording in progress, which a call of an :func:`eager_method` method abandons."""

    def __init__(self, fn: Callable[..., Any], device: wp.DeviceLike):
        """Initialize the callable.

        Args:
            fn: Callable to record. Its GPU work must be capturable; a method that is not is marked with
                :func:`eager_method`.
            device: Warp device that runs the callable.
        """
        self._fn = fn
        self._device = wp.get_device(device)
        self._recordings: dict[tuple, tuple[wp.Graph, Any]] = {}
        self._plain = False
        """Whether the callable called an :func:`eager_method` method while recording, so it runs plainly."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the callable and return its result."""
        # inside another capture (its kernels join it), on a CPU device, or after an abandoned recording: run plainly
        if capturing(self._device) or not self._device.is_cuda or self._plain:
            return self._fn(*args, **kwargs)
        # one recording per argument layout: array pointers and scalar values
        key = _argument_key(args, kwargs)
        recorded = self._recordings.get(key)
        if recorded is None:
            try:
                # the recording is the first run, so there is no eager warm-up whose work would run twice
                self._recordings[key] = recorded = self._record(args, kwargs)
            except _EagerCall as call:
                # the recording launched nothing; the callable cannot be recorded, so it runs plainly from now on,
                # starting with this call, until reset() lets it record again
                self._plain = True
                logger.warning(
                    "'%s' calls '%s', which cannot be recorded, so it runs without a CUDA graph.", _name(self._fn), call
                )
                return self._fn(*args, **kwargs)
            return recorded[1]
        graph, result = recorded
        # a replay writes into the buffers the recording call returned, so it returns that same result
        wp.capture_launch(graph)
        return result

    def reset(self) -> None:
        """Drop the recordings; the callable records again on its next call, also after it ran plainly."""
        self._plain = False
        if self._recordings:
            # graphs may still be executing; drain the device before releasing them
            wp.synchronize_device(self._device)
            self._recordings.clear()

    def _record(self, args: tuple, kwargs: dict[str, Any]) -> tuple[wp.Graph, Any]:
        """Record the callable as its first run and return the graph and the result.

        Raises:
            _EagerCall: If the callable calls an :func:`eager_method` method; the recording launched nothing.
            RuntimeError: If the callable runs work the graph cannot hold while it records (see :class:`_Recording`).
        """
        try:
            # the calls run once while they are recorded; their result is what every replay returns
            with _record_calls(self._device) as recording:
                result = self._fn(*args, **kwargs)
        except RuntimeError as exc:
            if not any(text in str(exc) for text in _UNRECORDABLE_ERRORS):
                raise
            # a host sync or an eager torch CUDA op inside the recording: name the fix instead of torch's message
            raise RuntimeError(
                f"'{_name(self._fn)}' synchronizes the host with the device or runs a torch CUDA operation outside the"
                " graph while it records, so a replay would keep the value it read or skip the operation. Run the"
                f" work in a method marked with @eager_method. ({exc})"
            ) from exc
        return recording.graph, result


@contextlib.contextmanager
def _record_calls(device: wp.Device):
    """Record the calls made inside as a CUDA graph; the recording is their first run.

    Yields:
        The recording, whose graph replays the calls.
    """
    recording = _Recording(device)
    enclosing, GraphedCallable._recording = GraphedCallable._recording, recording
    try:
        with wp.ScopedStream(recording.capture_stream):
            recording.begin()
            try:
                yield recording
            except BaseException:
                recording.abort()
                raise
            recording.end()
    finally:
        GraphedCallable._recording = enclosing


_UNRECORDABLE_ERRORS = (
    # torch, in sync debug mode "error", on a host synchronization
    "called a synchronizing CUDA operation",
    # CUDA, on work for the legacy default stream, torch's current stream, while a blocking Warp stream captures
    "operation would make the legacy stream depend on a capturing blocking stream",
    # CUDA, on a synchronization of the capturing stream
    "operation not permitted when stream is capturing",
    # the torch operation guard of a recording (see _TorchOpGuard)
    "runs outside the graph a recording captures",
)
"""Texts of the errors a call raises for work a graph cannot hold while it records."""


class _TorchOpGuard(TorchDispatchMode):
    """Raise on a torch operation on a device while a graph records there.

    Torch runs on its own stream, so the operation would run once at recording time and never in a replay. Views and
    other operations whose outputs alias their inputs launch no work and pass (``OpOverload.is_view``): wrapping Warp
    memory with :func:`warp.to_torch` (``aten.lift_fresh``), slicing, reshaping.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not func.is_view and self._touches_device(args, kwargs):
            raise RuntimeError(
                f"The torch operation {func} runs outside the graph a recording captures, once at recording time."
            )
        return func(*args, **kwargs)

    def _touches_device(self, args: tuple, kwargs: dict) -> bool:
        for leaf in _pytree.tree_leaves((args, kwargs)):
            device = leaf.device if isinstance(leaf, torch.Tensor) else leaf if isinstance(leaf, torch.device) else None
            # a device without an index, such as torch.device("cuda"), is the current one of its type
            if device is not None and device.type == self._device.type and device.index in (None, self._device.index):
                return True
        return False


class _Recording:
    """One recording: a CUDA graph, launched when it ends since the recording is the first run of the callable.

    Work the graph cannot hold raises while it records, and :meth:`GraphedCallable._record` names the call:

    * Torch runs on its current stream, which the recording does not change, so a torch operation on the recording's
      device would run once at recording time and never in a replay. While the graph records, a torch dispatch guard
      raises on every such operation that is not a view (see :class:`_TorchOpGuard`), with and without Kit. Without
      Kit, CUDA would reject it anyway (``cudaErrorStreamCaptureImplicit``: Warp captures a blocking stream, torch runs
      on the legacy one); with Kit, Warp captures a nonblocking stream in relaxed mode, which CUDA lets torch work
      bypass silently.
    * Torch runs in sync debug mode ``"error"`` (:func:`torch.cuda.set_sync_debug_mode`), so a host synchronization
      on any stream raises.
    * Warp raises on its own synchronizations during a capture.

    The guard and the previous sync debug mode are restored when the graph ends, also when the recording is abandoned.
    """

    def __init__(self, device: wp.Device):
        # the graph is captured on the capture stream and launched on the stream the callable runs on
        self.stream = wp.get_stream(device)
        self.capture_stream = self.stream
        self.mode = wp.CaptureMode.THREAD_LOCAL
        if has_kit():
            # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
            self.capture_stream = wp.stream_from_torch(torch.cuda.Stream(device=device.alias))
            self.mode = wp.CaptureMode.RELAXED
        self.graph: wp.Graph | None = None
        self.capturing = False
        # the sync debug mode outside the graph
        self._sync_debug_mode = torch.cuda.get_sync_debug_mode()
        self._torch_device = torch.device(wp.device_to_torch(device))
        self._guard: _TorchOpGuard | None = None

    def begin(self) -> None:
        """Start the graph."""
        wp.capture_begin(stream=self.capture_stream, capture_mode=self.mode)
        self.capturing = True
        self._enter_guards()

    def end(self) -> None:
        """End the graph and launch it."""
        self.capturing = False
        self._release()
        self.graph = wp.capture_end(stream=self.capture_stream)
        wp.capture_launch(self.graph, stream=self.stream)

    def abort(self) -> None:
        """End the graph without launching it, after the callable raised or called an :func:`eager_method` method."""
        if self.capturing:
            self.capturing = False
            self._release()
            with contextlib.suppress(Exception):
                wp.capture_end(stream=self.capture_stream)

    def _enter_guards(self) -> None:
        """Raise on torch work and host synchronizations while the graph records."""
        torch.cuda.set_sync_debug_mode("error")
        self._guard = _TorchOpGuard(self._torch_device)
        self._guard.__enter__()

    def _release(self) -> None:
        """Leave the torch guards of the graph that ends."""
        if self._guard is not None:
            guard, self._guard = self._guard, None
            guard.__exit__(None, None, None)
        torch.cuda.set_sync_debug_mode(self._sync_debug_mode)


def _argument_key(args: tuple, kwargs: dict[str, Any]) -> tuple:
    """Key a call by the pointer, shape, strides, dtype and device of its arrays and the type and value of its scalars.

    Positional and keyword arguments key apart.
    """
    return tuple(_value_key(value) for value in args), tuple(
        (name, _value_key(value)) for name, value in kwargs.items()
    )


def _value_key(value: Any) -> Any:
    # views of one allocation share a pointer, so the layout tells them apart
    if isinstance(value, wp.array):
        return value.ptr, value.shape, value.strides, value.dtype, str(value.device)
    if isinstance(value, torch.Tensor):
        return value.data_ptr(), value.shape, value.stride(), value.dtype, value.device
    # True, 1 and 1.0 compare equal, so the type tells them apart
    if value is None or isinstance(value, (bool, int, float, str)):
        return type(value), value
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_value_key(item) for item in value)
    return id(value)
