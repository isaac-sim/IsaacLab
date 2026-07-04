# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX-compatible CUDA graph capture helpers."""

from __future__ import annotations

import contextlib
import ctypes
import logging
from collections.abc import Callable

import warp as wp

logger = logging.getLogger(__name__)

# Relaxed capture lets RTX continue work on CUDA's legacy stream.
try:
    _cudart = ctypes.CDLL("libcudart.so.12")
except OSError:
    try:
        _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        _cudart = None


def capture_relaxed_graph(device: str, operation: Callable[[], None], *, warmup: bool = True):
    """Capture ``operation`` on an RTX-safe non-blocking CUDA stream.

    CUDA capture starts in relaxed mode before Warp registers the stream as an
    external capture. This avoids synchronization with RTX's legacy-stream
    work while allowing nested Warp conditional nodes. The returned Warp graph
    is patched with the raw graph handle because external capture does not yet
    expose that handle through Warp's public API.
    """
    if _cudart is None:
        logger.warning("libcudart not available; cannot use relaxed graph capture")
        return None

    if warmup:
        with wp.ScopedDevice(device):
            operation()
        wp.synchronize_stream(wp.get_stream(device))

    raw_handle = ctypes.c_void_p()
    ret = _cudart.cudaStreamCreateWithFlags(ctypes.byref(raw_handle), ctypes.c_uint(0x01))
    if ret != 0:
        logger.warning("cudaStreamCreateWithFlags(NonBlocking) failed (code %d)", ret)
        return None
    fresh_handle = raw_handle.value
    fresh_stream = wp.Stream(device, cuda_stream=fresh_handle, owner=False)

    ret = _cudart.cudaStreamBeginCapture(ctypes.c_void_p(fresh_handle), ctypes.c_int(2))
    if ret != 0:
        _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))
        logger.warning("cudaStreamBeginCapture(relaxed) failed (code %d)", ret)
        return None

    try:
        wp.capture_begin(stream=fresh_stream, external=True)
    except Exception as exc:
        raw_graph = ctypes.c_void_p()
        _cudart.cudaStreamEndCapture(ctypes.c_void_p(fresh_handle), ctypes.byref(raw_graph))
        if raw_graph.value:
            _cudart.cudaGraphDestroy(raw_graph)
        _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))
        logger.warning("wp.capture_begin(external=True) failed: %s", exc)
        return None

    error = None
    with wp.ScopedStream(fresh_stream, sync_enter=False):
        try:
            operation()
        except Exception as exc:
            error = exc

    if error is None:
        try:
            graph = wp.capture_end(stream=fresh_stream)
        except Exception as exc:
            error = exc
            graph = None
    else:
        with contextlib.suppress(Exception):
            wp.capture_end(stream=fresh_stream)
        graph = None

    raw_graph = ctypes.c_void_p()
    end_ret = _cudart.cudaStreamEndCapture(ctypes.c_void_p(fresh_handle), ctypes.byref(raw_graph))
    _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))

    if error is not None:
        if raw_graph.value:
            _cudart.cudaGraphDestroy(raw_graph)
        logger.warning("Newton graph capture aborted: %s", error)
        return None
    if end_ret != 0 or not raw_graph.value:
        logger.warning("cudaStreamEndCapture failed (code %d)", end_ret)
        return None

    graph.graph = raw_graph
    graph.graph_exec = None
    return graph
