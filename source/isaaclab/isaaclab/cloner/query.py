# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched numeric topology queries. Resolve declared paths separately in :mod:`.path`.

All queries return ``(world_indices, world_starts)``. Indices are flat int32 values;
starts are int64 offsets with shape [num_queries, num_worlds + 2], including shared world -1.
For query q and world w, ``world_starts[q, w + 1 : w + 3]`` bounds its selected instances.
Each row's first/last offset bounds the entire query. Repeated IDs retain separate results.

NumPy queries allocate exact-sized results. Warp queries require resident int32 query IDs
and preallocated ``out`` arrays on the topology's device; no upload or readback is implicit.
Warm up before CUDA graph capture. For nonempty batches, the valid prefix ends at ``world_starts[-1, -1]``.
If that required size exceeds capacity, starts are still reported but indices are left untouched:
the caller must provide enough capacity for its selection domain, not consume a partial result.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .clone_plan import PrototypeWorldTopology


def get_asset_prototype_world_index(
    topology: PrototypeWorldTopology,
    asset_prototype: int | np.ndarray | wp.array,
    *,
    out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
    """Return one world index per asset instance, retaining repeated memberships.

    Args:
        topology: Numeric topology with NumPy or Warp storage.
        asset_prototype: One integer (NumPy only), or a 1-D integer array of asset-prototype IDs.
        out: Optional NumPy outputs, required Warp outputs. See the module's result/capacity contract.

    Returns:
        Flat world indices and per-query world boundaries. A scalar is a batch of length one.
        Missing or unused asset IDs produce empty slices; shared instances use world -1.
    """
    return _world_index(topology, asset_prototype, by_asset=True, unique=False, out=out)


def get_asset_prototype_unique_world_index(
    topology: PrototypeWorldTopology,
    asset_prototype: int | np.ndarray | wp.array,
    *,
    out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
    """Return each world containing an asset once, independently for every requested asset.

    Args:
        topology: Numeric topology with NumPy or Warp storage.
        asset_prototype: One integer (NumPy only), or a 1-D integer array of asset-prototype IDs.
        out: Optional NumPy outputs, required Warp outputs. See the module's result/capacity contract.

    Returns:
        Flat world indices and per-query world boundaries, as in :func:`get_asset_prototype_world_index`.
        Every world slice has length zero or one. Separate queries are not deduplicated together.
    """
    return _world_index(topology, asset_prototype, by_asset=True, unique=True, out=out)


def get_world_prototype_world_index(
    topology: PrototypeWorldTopology,
    world_prototype: int | np.ndarray | wp.array,
    *,
    out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
    """Return the destination worlds using each requested world prototype.

    Args:
        topology: Numeric topology with NumPy or Warp storage.
        world_prototype: One integer (NumPy only), or a 1-D integer array of world-prototype IDs.
            Index -1 selects the shared world, even when empty.
        out: Optional NumPy outputs, required Warp outputs. See the module's result/capacity contract.

    Returns:
        Flat world indices and per-query world boundaries, as in :func:`get_asset_prototype_world_index`.
        Unused world prototypes produce empty slices.
    """
    return _world_index(topology, world_prototype, by_asset=False, unique=True, out=out)


def _world_index(
    topology: PrototypeWorldTopology,
    prototype_ids: int | np.ndarray | wp.array,
    *,
    by_asset: bool,
    unique: bool,
    out: tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array] | None,
) -> tuple[np.ndarray, np.ndarray] | tuple[wp.array, wp.array]:
    num_worlds = len(topology.world_prototype_layout)
    if isinstance(topology.world_prototypes, np.ndarray):
        if isinstance(prototype_ids, wp.array):
            raise TypeError("Warp query IDs require to_warp(topology, device); queries do not transfer arrays.")
        prototype_ids = np.atleast_1d(prototype_ids)
        if prototype_ids.ndim != 1 or prototype_ids.dtype.kind not in "iu":
            raise TypeError("Topology queries require integer IDs; resolve expressions with cloner.path.")
        layout = np.r_[-1, topology.world_prototype_layout]
        if by_asset:
            matches = prototype_ids[:, None] == topology.world_prototypes
            prefix = np.zeros((len(prototype_ids), len(topology.world_prototypes) + 1), dtype=np.int64)
            np.cumsum(matches, axis=1, out=prefix[:, 1:])
            counts = np.diff(prefix[:, topology.world_prototype_starts], axis=1)[:, layout + 1]
            if unique:
                counts = counts > 0
        else:
            counts = prototype_ids[:, None] == layout
        starts = np.zeros((len(prototype_ids), num_worlds + 2), dtype=np.int64)
        starts[:, 1:] = counts
        np.cumsum(starts.ravel(), out=starts.ravel())
        indices = np.repeat(np.tile(np.arange(-1, num_worlds, dtype=np.int32), len(prototype_ids)), counts.ravel())
        if out is None:
            return indices, starts
        out[1][:] = starts
        if len(indices) <= len(out[0]):
            out[0][: len(indices)] = indices
    else:
        if out is None:
            raise ValueError("Warp queries require preallocated out=(world_indices, world_starts).")
        indices, starts = out
        if starts.shape != (len(prototype_ids), num_worlds + 2) or not starts.is_contiguous:
            raise ValueError("world_starts must be contiguous with shape [num_queries, num_worlds + 2].")
        wp.launch(
            _count_world_instances,
            dim=starts.shape,
            inputs=[
                topology.world_prototypes,
                topology.world_prototype_starts,
                topology.world_prototype_layout,
                prototype_ids,
                by_asset,
                unique,
            ],
            outputs=[starts],
            device=starts.device,
        )
        wp.utils.array_scan(starts.flatten(), starts.flatten())
        wp.launch(_fill_world_indices, (len(prototype_ids), num_worlds + 1), [starts, indices], device=starts.device)
    return out


@wp.kernel
def _count_world_instances(
    world_prototypes: wp.array(dtype=wp.int32),
    world_prototype_starts: wp.array(dtype=wp.int64),
    world_prototype_layout: wp.array(dtype=wp.int32),
    prototype_ids: wp.array(dtype=wp.int32),
    by_asset: bool,
    unique: bool,
    starts: wp.array2d(dtype=wp.int64),
):
    query, column = wp.tid()
    count = wp.int64(0)
    if column > 0:
        world_prototype = -1
        if column > 1:
            world_prototype = world_prototype_layout[column - 2]
        if by_asset:
            member = world_prototype_starts[world_prototype + 1]
            end = world_prototype_starts[world_prototype + 2]
            while member < end:
                if world_prototypes[member] == prototype_ids[query]:
                    count += wp.int64(1)
                member += wp.int64(1)
            if unique:
                count = wp.min(count, wp.int64(1))
        elif world_prototype == prototype_ids[query]:
            count = wp.int64(1)
    starts[query, column] = count


@wp.kernel
def _fill_world_indices(starts: wp.array2d(dtype=wp.int64), indices: wp.array(dtype=wp.int32)):
    query, world = wp.tid()
    # Never write a partial result or past capacity, including during graph replay.
    if starts[starts.shape[0] - 1, starts.shape[1] - 1] <= wp.int64(indices.shape[0]):
        index = starts[query, world]
        while index < starts[query, world + 1]:
            indices[index] = world - 1
            index += wp.int64(1)
