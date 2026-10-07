# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Tracy profiling of the task, its solvers and the RTX renderer.

OVRTX ships Carbonite's profiler with its Tracy backend. :func:`enable` configures it to start with the renderer, whose
CPU zones cover stepping, scene population and RTX rendering, and whose GPU zones time the RTX render passes. The task
adds its own CPU zones (frames, physics, coupled solvers, Gaussian upload, renderer calls) through the same Tracy
client, so that one capture holds all of them. Zones open before the renderer starts are not recorded, and CUDA work
(the physics and the Gaussian deformation) has no GPU zones.

Connect with a Tracy viewer whose protocol matches the bundled client, such as the ``Tracy`` or ``capture`` binaries of
Kit's ``omni.kit.profiler.tracy`` 1.2 extension; the client listens on port 8086.
"""

import contextlib
import ctypes
import os
from collections.abc import Iterable
from pathlib import Path

import warp as wp

PROFILER_SETTINGS = (
    "--/app/profilerBackend=tracy",
    "--/app/profileFromStart=true",
    # Channel 2 adds detail zones, mostly in scene population and geometry updates.
    "--/app/profilerMask=3",
    "--/plugins/carb.profiler-tracy.plugin/fibersAsThreads=false",
    "--/plugins/carb.profiler-tracy.plugin/instantEventsAsMessages=true",
    "--/profiler/gpu/tracyInject/enabled=true",
    "--/rtx/addTileGpuAnnotations=true",
)
"""Carbonite settings that start the renderer's profiler with its Tracy backend and GPU zones.

OVRTX's public ``enable_profiling`` option starts the same profiler but forces the capture mask to 1.
"""

_enabled = False
_sync = False
_client = None
_locations = {}


class _SourceLocation(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("function", ctypes.c_char_p),
        ("file", ctypes.c_char_p),
        ("line", ctypes.c_uint32),
        ("color", ctypes.c_uint32),
    ]


class _ZoneContext(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("active", ctypes.c_int32)]


def enable(sync: bool = False, settings: Iterable[str] = ()) -> None:
    """Profile with Tracy; call before the first OVRTX renderer is created.

    Args:
        sync: Synchronize the device at the end of each task zone, so that zones measure GPU work rather than its
            launch. This serializes CPU and GPU, and the physics needs CUDA graphs off for its solver zones to appear
            on every step.
        settings: Additional Carbonite settings for the renderer, such as ``--/app/profilerMask=7``; they override
            :data:`PROFILER_SETTINGS`.
    """
    global _enabled, _sync
    _apply_renderer_settings((*PROFILER_SETTINGS, *settings))
    _enabled, _sync = True, sync


def enabled() -> bool:
    """Return whether Tracy profiling is enabled."""
    return _enabled


def _apply_renderer_settings(settings: Iterable[str]) -> None:
    """Apply Carbonite settings to OVRTX before its first renderer, through its internal settings extension.

    OVRTX reads no settings files or setting environment variables; the ``ovrtx-extensions`` package wraps the same
    native extension.
    """
    import ovrtx
    from ovrtx._src.bindings import ovrtx_result_t, ovx_string_t

    class ApplySettings(ctypes.Structure):
        _fields_ = [("apply_settings", ctypes.CFUNCTYPE(ovrtx_result_t, ovx_string_t))]

    # The OVRTX package loads this same library when it creates its first renderer.
    library = ctypes.CDLL(str(Path(ovrtx.__file__).parent / "bin" / "libovrtx-dynamic.so"))
    library.ovrtx_query_extension.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    library.ovrtx_query_extension.restype = ovrtx_result_t
    table = ctypes.c_void_p()
    if library.ovrtx_query_extension(b"ovrtx.settings.apply_settings", ctypes.byref(table)).status.value != 0:
        raise RuntimeError("This OVRTX build has no settings extension: Tracy profiling is unavailable")
    apply = ctypes.cast(table, ctypes.POINTER(ApplySettings)).contents.apply_settings
    for setting in settings:
        if apply(ovx_string_t(setting)).status.value != 0:
            raise ValueError(f"OVRTX rejected the setting {setting!r}")


class _Client:
    """The Tracy C API of the profiler plugin that OVRTX loaded."""

    def __init__(self, library: ctypes.CDLL):
        def function(name, argtypes, restype):
            call = getattr(library, f"___tracy_{name}")
            call.argtypes, call.restype = argtypes, restype
            return call

        self.begin = function("emit_zone_begin", [ctypes.POINTER(_SourceLocation), ctypes.c_int32], _ZoneContext)
        self.end = function("emit_zone_end", [_ZoneContext], None)
        self.frame_mark = function("emit_frame_mark", [ctypes.c_char_p], None)
        self.started = function("profiler_started", [], ctypes.c_int32)
        self.set_thread_name = function("set_thread_name", [ctypes.c_char_p], None)
        self.thread_named = False


def _tracy() -> _Client | None:
    """Return the Tracy client once OVRTX has loaded and started it, or None."""
    global _client
    if _client is None:
        import ovrtx

        path = Path(ovrtx.__file__).parent / "bin" / "plugins" / "libcarb.profiler-tracy.plugin.so"
        try:
            # Bind the instance OVRTX loaded, never a second one.
            _client = _Client(ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_GLOBAL))
        except OSError:
            return None
    if not _client.started():
        return None
    if not _client.thread_named:
        _client.set_thread_name(b"Python main")
        _client.thread_named = True
    return _client


@contextlib.contextmanager
def zone(name: str, color: int = 0):
    """Record a Tracy zone around a block, as a no-op while profiling is disabled or not started.

    Args:
        name: Zone name.
        color: Zone color as ``0xRRGGBB``; 0 uses the viewer's default.
    """
    client = _tracy() if _enabled else None
    if client is None:
        yield
        return
    location = _locations.get((name, color))
    if location is None:
        # Tracy keeps a pointer to the source location: keep it, and its strings, alive.
        location = _locations[(name, color)] = _SourceLocation(name.encode(), name.encode(), b"python", 0, color)
    context = client.begin(ctypes.byref(location), 1)
    try:
        yield
    finally:
        if _sync:
            wp.synchronize()
        client.end(context)


def frame_mark() -> None:
    """Mark the end of an application frame."""
    client = _tracy() if _enabled else None
    if client is not None:
        client.frame_mark(None)


def instrument_solvers(coupled) -> None:
    """Record a zone for each step of a coupled solver and of its entry solvers.

    The zones are recorded when the step runs in Python: with CUDA graphs, only while the graph is captured.
    """
    instrument(coupled, "step", "coupled solver", 0x4C8BF5)
    for entry in coupled.entry_names():
        solver = coupled.solver(entry)
        instrument(solver, "step", f"{entry}: {type(solver).__name__}", 0x34A853)


def instrument(target, method: str, name: str, color: int = 0) -> None:
    """Record a zone around each call of an object's method.

    Args:
        target: Object whose method to wrap; only this instance is affected.
        method: Method name.
        name: Zone name.
        color: Zone color as ``0xRRGGBB``; 0 uses the viewer's default.
    """
    call = getattr(target, method)

    def profiled(*args, **kwargs):
        with zone(name, color):
            return call(*args, **kwargs)

    setattr(target, method, profiled)
