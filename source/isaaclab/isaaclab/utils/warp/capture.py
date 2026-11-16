# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph recording of the Warp frontend: :class:`GraphedCallable` records a call as a CUDA graph and replays it.
A call that cannot be recorded goes through :func:`eager`; a recorded call that makes one runs plainly instead.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.utils._pytree as _pytree
import warp as wp
import warp._src.context as _warp_context
from torch.utils._python_dispatch import TorchDispatchMode

from ..version import has_kit
from .launch_cache import _WarpLaunchCache

if TYPE_CHECKING:
    from ...physics import PhysicsManager

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


def eager(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function that cannot be recorded.

    Outside a recording this is a plain call. A :class:`GraphedCallable` that meets it while recording abandons the
    recording, which launches nothing, and runs plainly from then on, this call included.

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
        raise _EagerCall(fn)
    if capturing():
        raise RuntimeError(
            f"'{_name(fn)}' cannot be recorded, so it cannot run inside a graph capture that no GraphedCallable"
            " started."
        )
    return fn(*args, **kwargs)


class _EagerCall(BaseException):
    """Raised by :func:`eager` inside a recording, which the :class:`GraphedCallable` abandons.

    It derives from :class:`BaseException`, so the ``except Exception`` handlers of the recorded code let it through.
    """

    def __init__(self, fn: Callable[..., Any]):
        super().__init__(_name(fn))


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__qualname__", type(fn).__name__)


def eager_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a method that cannot be recorded through :func:`eager`.

    A :class:`GraphedCallable` that calls the method runs plainly. An override in a subclass is decorated again.

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
    """A callable recorded as a CUDA graph on its first call with a given argument layout and replayed after.

    * The first call with a given argument layout records without an eager warm-up: the recording is its first
      execution. Later calls with the same array pointers and scalar values replay that recording and return the
      result of the recording call, so the callable returns buffers its owner keeps.
    * Calls that cannot be recorded go through :func:`eager`. A recording that meets one is abandoned without
      launching anything, and the callable runs plainly from then on, starting with this call; it records again after
      :meth:`reset`. Known limitation: the host work the callable did before the :func:`eager` call, such as a Python
      counter it advanced, ran once in the abandoned recording and runs again in the plain call, on that first call
      only; its device work ran only in the plain call.
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
    """The recording in progress, which an :func:`eager` call abandons."""

    def __init__(self, fn: Callable[..., Any], device: wp.DeviceLike):
        """Initialize the callable.

        Args:
            fn: Callable to record. Its GPU work must be capturable; a call that is not goes through :func:`eager`.
            device: Warp device that runs the callable.
        """
        self._fn = fn
        self._device = wp.get_device(device)
        self._recordings: dict[tuple, tuple[wp.Graph, Any]] = {}
        self._plain = False
        """Whether the callable met an :func:`eager` call while recording, so it runs plainly."""
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
        if self._plain:
            return self._fn(*args, **kwargs)
        key = _argument_key(args, kwargs)
        recorded = self._recordings.get(key)
        if recorded is None:
            try:
                self._recordings[key] = recorded = self._record(args, kwargs)
            except _EagerCall as call:
                self._plain = True
                logger.warning(
                    "'%s' calls '%s', which cannot be recorded, so it runs without a CUDA graph.", _name(self._fn), call
                )
                return self._fn(*args, **kwargs)
            return recorded[1]
        graph, result = recorded
        wp.capture_launch(graph)
        return result

    def reset(self) -> None:
        """Drop the recordings; the callable records again on its next call, also after it ran plainly."""
        self._plain = False
        if self._recordings:
            # graphs may still be executing; drain the device before releasing them
            wp.synchronize_device(self._device)
            self._recordings.clear()

    @classmethod
    def watch(cls, backend_manager: PhysicsManager) -> None:
        """Record every callable and every cached launch again after the physics backend rebinds its buffers or stops,
        and settle the backend before each recording: work it queued before (model changes, kinematics) runs then,
        once, so a recording holds only the work of its own calls.

        Args:
            backend_manager: Physics manager that dispatches the lifecycle events.
        """
        # imported here so that the utilities do not load the physics package
        from ...physics import PhysicsEvent

        cls.unwatch()
        cls._settle = backend_manager.forward
        cls._callback_handles = [
            backend_manager.register_callback(callback, event)
            for event in (PhysicsEvent.PHYSICS_READY, PhysicsEvent.STOP)
            # the recorded launches of the code outside the graphs read the same buffers
            for callback in (cls.invalidate, _WarpLaunchCache.clear_all)
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

    def _record(self, args: tuple, kwargs: dict[str, Any]) -> tuple[wp.Graph, Any]:
        """Record the callable as its first run and return the graph and the result.

        Raises:
            _EagerCall: If the callable makes an :func:`eager` call; the recording launched nothing.
            RuntimeError: If the callable runs work the graph cannot hold while it records (see :class:`_Recording`).
        """
        try:
            with _record_calls(self._device) as recording:
                result = self._fn(*args, **kwargs)
        except RuntimeError as exc:
            if not any(text in str(exc) for text in _UNRECORDABLE_ERRORS):
                raise
            raise RuntimeError(
                f"'{_name(self._fn)}' synchronizes the host with the device or runs a torch CUDA operation outside the"
                " graph while it records, so a replay would keep the value it read or skip the operation. Run the"
                f" function through eager() or mark it with @eager_method. ({exc})"
            ) from exc
        return recording.graph, result


@contextlib.contextmanager
def _record_calls(device: wp.Device):
    """Record the calls made inside as a CUDA graph; the recording is their first run.

    Yields:
        The recording, whose graph replays the calls.
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
        """End the graph without launching it, after the callable raised or made an :func:`eager` call."""
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
