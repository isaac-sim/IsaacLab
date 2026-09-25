# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene queries on explicit Newton resources."""

from __future__ import annotations

import contextlib
import ctypes
import gc
import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.version import has_kit

if TYPE_CHECKING:
    from ..physics.newton_manager import NewtonBackend

logger = logging.getLogger(__name__)

try:
    _cudart = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}") if torch.version.cuda else None
except OSError:
    _cudart = None


@contextlib.contextmanager
def _paused_gc():
    """Keep collection-driven frees out of CUDA graph capture."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _refit_bvh(backend: NewtonBackend) -> None:
    model, state = backend.model, backend.state_0
    if model.shape_count:
        model.bvh_refit_shapes(state)
    if model.particle_count:
        model.bvh_refit_particles(state)


def run_query(
    backend: NewtonBackend,
    timestamp: tuple[int, int],
    query: Callable[[], None],
    graph: tuple[tuple[int, ...], wp.Graph] | None,
    *,
    use_cuda_graph: bool = True,
) -> tuple[tuple[int, ...], wp.Graph] | None:
    """Refresh the shared BVHs once per publication and run a consumer-owned query graph.

    Args:
        backend: Shared model, state, and acceleration structures.
        timestamp: Published transform and geometry timestamps for the supplied state.
        query: Camera or ray-cast work using the supplied backend.
        graph: This consumer's previous captured query and pointer layout, or None.
        use_cuda_graph: Whether to capture GPU queries. CPU queries always run eagerly.

    Returns:
        The consumer's captured query and pointer layout, or None for eager execution.
    """
    model, state = backend.model, backend.state_0
    pointers = tuple(array.ptr if array is not None else 0 for array in (state.body_q, state.particle_q))
    if backend.bvh_timestamp != timestamp:
        if model.shape_count and model.bvh_shapes is None:
            model.bvh_build_shapes(state)
        if model.particle_count and model.bvh_particles is None:
            model.bvh_build_particles(state)
        if model.device.is_cuda and use_cuda_graph:
            if backend.bvh_graph is None or backend.bvh_graph[0] != pointers:
                _refit_bvh(backend)
                backend.bvh_graph = (
                    pointers,
                    capture_graph(str(model.device), partial(_refit_bvh, backend), relaxed=has_kit()),
                )
            wp.capture_launch(backend.bvh_graph[1])
        else:
            _refit_bvh(backend)
        backend.bvh_timestamp = timestamp
    if not model.device.is_cuda or not use_cuda_graph:
        query()
        return None
    if graph is None or graph[0] != pointers:
        if not has_kit():
            query()  # Allocate the consumer's scratch data before capture.
        graph = pointers, capture_graph(str(model.device), query, relaxed=has_kit())
    wp.capture_launch(graph[1])
    return graph


def capture_graph(device: str, capture_target: Callable[[], None], *, relaxed: bool = False):
    """Capture explicit work; relaxed mode isolates Kit's background CUDA streams."""
    if not relaxed:
        with wp.ScopedCapture(device=device) as capture:
            capture_target()
        return capture.graph
    if _cudart is None:
        logger.warning("libcudart not available; cannot use relaxed graph capture")
        return None

    # Warmup: pre-allocate all solver scratch buffers so the capture window has
    # no new cudaMalloc calls (which are forbidden inside graph capture).
    simulate = capture_target
    with wp.ScopedDevice(device):
        simulate()
    wp.synchronize_stream(wp.get_stream(device))

    # Create a non-blocking stream (cudaStreamNonBlocking = 0x01).
    raw_handle = ctypes.c_void_p()
    ret = _cudart.cudaStreamCreateWithFlags(ctypes.byref(raw_handle), ctypes.c_uint(0x01))
    if ret != 0:
        logger.warning("cudaStreamCreateWithFlags(NonBlocking) failed (code %d)", ret)
        return None
    fresh_handle = raw_handle.value
    fresh_stream = wp.Stream(device, cuda_stream=fresh_handle, owner=False)

    with _paused_gc():
        # Start capture in relaxed mode BEFORE entering ScopedStream.
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

        err_during_capture = None
        with wp.ScopedStream(fresh_stream, sync_enter=False):
            try:
                simulate()
            except Exception as exc:
                err_during_capture = exc

        if err_during_capture is None:
            try:
                graph = wp.capture_end(stream=fresh_stream)
            except Exception as exc:
                err_during_capture = exc
                graph = None
        else:
            with contextlib.suppress(Exception):
                wp.capture_end(stream=fresh_stream)
            graph = None

        raw_graph = ctypes.c_void_p()
        end_ret = _cudart.cudaStreamEndCapture(ctypes.c_void_p(fresh_handle), ctypes.byref(raw_graph))
        _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))

    if err_during_capture is not None:
        if raw_graph.value:
            _cudart.cudaGraphDestroy(raw_graph)
        logger.warning("Newton graph capture aborted during simulate: %s", err_during_capture)
        return None

    if end_ret != 0 or not raw_graph.value:
        logger.warning("cudaStreamEndCapture failed (code %d)", end_ret)
        return None

    # Patch the Warp Graph object with the raw CUDA graph handle obtained
    # from our external cudaStreamEndCapture.  wp.capture_end(external=True)
    # returns a Graph with a stale handle; we overwrite it so that
    # wp.capture_launch() replays the correct graph.
    # NOTE: This relies on Warp internals (Graph.graph / Graph.graph_exec).
    # Setting graph_exec = None triggers lazy cudaGraphInstantiate on
    # the next capture_launch.  Replace with public API when available.
    graph.graph = raw_graph
    graph.graph_exec = None
    return graph
