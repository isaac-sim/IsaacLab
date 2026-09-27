# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene queries on explicit Newton resources."""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.version import has_kit

if TYPE_CHECKING:
    from ..physics.newton_manager import NewtonBackend


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
            gc.collect()


def _refit_bvh(backend: NewtonBackend) -> None:
    model, state = backend.model, backend.state_0
    if model.shape_count:
        model.bvh_refit_shapes(state)
    if model.particle_count:
        model.bvh_refit_particles(state)


def run_query(
    backend: NewtonBackend,
    timestamp: int,
    query: Callable[[], None],
    graph: tuple[tuple[int, ...], wp.Graph] | None,
    *,
    use_cuda_graph: bool = True,
) -> tuple[tuple[int, ...], wp.Graph] | None:
    """Refresh the shared BVHs once per publication and run a consumer-owned query graph.

    Args:
        backend: Shared model, state, and acceleration structures.
        timestamp: Sum of the monotonic SDP transform and geometry timestamps for this state.
        query: Camera or ray-cast work using the supplied backend.
        graph: This consumer's previous captured query and pointer layout, or None.
        use_cuda_graph: Whether to capture GPU queries. CPU queries always run eagerly.

    Returns:
        The consumer's captured query and pointer layout, or None for eager execution.
    """
    model, state = backend.model, backend.state_0
    pointers = tuple(array.ptr if array is not None else 0 for array in (state.body_q, state.particle_q))
    cached = backend.bvh_refit
    if cached.timestamp != timestamp:
        if model.device.is_cuda and use_cuda_graph:
            if cached.data is None or cached.data[0] != pointers:
                cached.data = (
                    pointers,
                    capture_graph(str(model.device), partial(_refit_bvh, backend), relaxed=has_kit()),
                )
            wp.capture_launch(cached.data[1])
        else:
            _refit_bvh(backend)
        cached.timestamp = timestamp
    if not model.device.is_cuda or not use_cuda_graph:
        query()
        return None
    if graph is None or graph[0] != pointers:
        graph = pointers, capture_graph(str(model.device), query, relaxed=has_kit())
    wp.capture_launch(graph[1])
    return graph


def capture_graph(device: str, capture_target: Callable[[], None], *, relaxed: bool = False) -> wp.Graph:
    """Record work without an eager warmup or graph replay, using Warp's public capture API."""
    stream = wp.get_stream(device)
    mode = wp.CaptureMode.THREAD_LOCAL
    if relaxed:
        # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
        stream = wp.stream_from_torch(torch.cuda.Stream(device=device))
        mode = wp.CaptureMode.RELAXED
    with _paused_gc(), wp.ScopedStream(stream):
        with wp.ScopedCapture(stream=stream, capture_mode=mode) as capture:
            capture_target()
    return capture.graph
