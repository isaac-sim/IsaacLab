# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load recorded trajectory datasets into host memory for training."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from physicsnemo_robotics.data import InMemoryTrajectoryReader, NewtonTrajectoryZarrReader
from physicsnemo_robotics.dynamics import TrajectoryMetadata, TrajectorySchema
from tensordict import TensorDict

_CHUNK_TRAJECTORIES = 1024
"""Trajectories read by one worker task."""

_CACHE_WAIT_S = 4 * 3600.0
"""Longest time a data-parallel rank waits for rank zero to write the dataset cache [s]."""


def expand_dataset_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve dataset paths, replacing each directory of ``shard-*`` datasets by its shards.

    Args:
        paths: One dataset, several datasets, or directories of shards written by
            :class:`~isaaclab_physicsnemo.TrajectoryDatasetRecorder` with ``shard_episodes``.

    Returns:
        The resolved dataset directories in a deterministic order.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved = [Path(path).expanduser().resolve() for path in paths]
    return [shard for path in resolved for shard in (sorted(path.glob("shard-*")) or [path])]


def load_trajectory_datasets(
    paths: Sequence[Path],
    *,
    workers: int = 1,
    cache_path: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> InMemoryTrajectoryReader:
    """Read datasets of one schema into a single in-memory reader, in path and manifest order.

    Recorded trajectories are separate Zarr groups, so reading them is dominated by per-file overhead; worker
    processes read chunks of trajectories in parallel. A cache stores every trajectory in one file, which later
    loads read in seconds. Under data parallelism, rank zero reads the datasets and writes the cache, and the other
    ranks load the cache instead of reading every group again.

    Args:
        paths: Dataset directories, see :func:`expand_dataset_paths`.
        workers: Worker processes that read trajectories. One reads them in the calling process.
        cache_path: File that caches the loaded trajectories. It is reused while the datasets are unchanged.
            Required when ``world_size`` is greater than one.
        rank: Data-parallel rank of the caller.
        world_size: Number of data-parallel ranks.

    Returns:
        The in-memory reader with the trajectories and metadata of every dataset.

    Raises:
        ValueError: If the datasets differ in schema or share trajectories, or several ranks load without a cache.
        TimeoutError: If rank zero does not write the cache in time.
    """
    if world_size > 1 and cache_path is None:
        raise ValueError("Data-parallel loading shares the trajectories through a cache file; set cache_path.")
    key = _cache_key(paths)
    if cache_path is not None:
        cached = _read_cache(cache_path, key)
        if cached is None and rank != 0:
            # Rank zero replaces a stale cache, so wait for one that matches these datasets.
            deadline = time.monotonic() + _CACHE_WAIT_S
            while cached is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Rank zero did not write the dataset cache {cache_path} in time.")
                time.sleep(10.0)
                cached = _read_cache(cache_path, key)
        if cached is not None:
            return cached

    tasks, schema = [], None
    for path in paths:
        # The reader returned by ``with`` is typed as the generic base reader, so keep the concrete one.
        stored = NewtonTrajectoryZarrReader(path)
        with stored:
            if schema is not None and stored.schema.differences(schema):
                raise ValueError(f"The dataset at {path} has a different schema than {paths[0]}.")
            schema, count = stored.schema, len(stored)
        tasks += [
            (path, start, min(start + _CHUNK_TRAJECTORIES, count)) for start in range(0, count, _CHUNK_TRAJECTORIES)
        ]
    if workers > 1 and len(tasks) > 1:
        # Forked workers only read files into CPU tensors, so they neither touch the parent's CUDA context nor
        # require the calling script to guard its entry point, as spawned workers would.
        context = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
        with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=context) as pool:
            chunks = list(pool.map(_read_chunk, tasks))
    else:
        chunks = [_read_chunk(task) for task in tasks]

    fields = [item for values, lengths, _, _ in chunks for item in _unpack(values, lengths)]
    metadata = [TrajectoryMetadata.from_dict(item) for _, _, chunk_metadata, _ in chunks for item in chunk_metadata]
    if len({item.trajectory_id for item in metadata}) != len(metadata):
        raise ValueError("The datasets share trajectory identifiers; record them with different seeds.")
    batch_size = torch.Size(chunks[0][3])
    reader = InMemoryTrajectoryReader(
        [TensorDict(item, batch_size=batch_size) for item in fields], schema=schema, metadata=metadata
    )
    if cache_path is not None and rank == 0:
        _write_cache(cache_path, key, schema, fields, metadata, batch_size)
    return reader


def _read_chunk(
    task: tuple[Path, int, int],
) -> tuple[dict[str, np.ndarray], list[dict[str, int]], list[dict[str, Any]], tuple[int, ...]]:
    """Read trajectories ``[start, stop)`` of one dataset as one array per field and metadata dictionaries.

    Arrays travel back to the parent by value. Returning one Torch tensor per trajectory field would map a shared
    memory segment for each, which exhausts the process's memory maps for a hundred thousand trajectories.
    """
    path, start, stop = task
    stored = NewtonTrajectoryZarrReader(path)
    with stored:
        trajectories = [stored[index][0] for index in range(start, stop)]
        metadata = [stored.trajectory_metadata(index).to_dict() for index in range(start, stop)]
    values, lengths = _pack([dict(trajectory.items()) for trajectory in trajectories])
    return {name: value.numpy() for name, value in values.items()}, lengths, metadata, tuple(trajectories[0].batch_size)


def _pack(fields: list[dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], list[dict[str, int]]]:
    """Concatenate each field over the trajectories and keep the per-trajectory lengths."""
    names = list(fields[0])
    lengths = [{name: item[name].shape[0] for name in names} for item in fields]
    return {name: torch.cat([item[name] for item in fields]) for name in names}, lengths


def _unpack(
    values: Mapping[str, torch.Tensor | np.ndarray], lengths: list[dict[str, int]]
) -> list[dict[str, torch.Tensor]]:
    """Split concatenated fields back into per-trajectory views."""
    tensors = {name: torch.as_tensor(value) for name, value in values.items()}
    offsets = dict.fromkeys(tensors, 0)
    fields = []
    for trajectory in lengths:
        item = {}
        for name, length in trajectory.items():
            item[name] = tensors[name][offsets[name] : offsets[name] + length]
            offsets[name] += length
        fields.append(item)
    return fields


def _cache_key(paths: Sequence[Path]) -> list[tuple[str, int]]:
    """Identify datasets by path and directory modification time, which changes when an episode is added."""
    return [(str(path), os.stat(path).st_mtime_ns) for path in paths]


def _key_path(cache_path: Path) -> Path:
    """Small file naming the datasets of a cache, which waiting ranks poll instead of the cache itself."""
    return cache_path.with_name(f"{cache_path.name}.key.json")


def _read_cache(cache_path: Path, key: list[tuple[str, int]]) -> InMemoryTrajectoryReader | None:
    """Load cached trajectories, or return ``None`` if the cache is missing or belongs to other datasets."""
    key_path = _key_path(cache_path)
    if not key_path.is_file() or json.loads(key_path.read_text()) != [list(item) for item in key]:
        return None
    cache = torch.load(cache_path, weights_only=False)
    schema = TrajectorySchema.from_dict(cache["schema"])
    metadata = [TrajectoryMetadata.from_dict(item) for item in cache["metadata"]]
    trajectories = [
        TensorDict(item, batch_size=cache["batch_size"]) for item in _unpack(cache["values"], cache["lengths"])
    ]
    return InMemoryTrajectoryReader(trajectories, schema=schema, metadata=metadata)


def _write_cache(
    cache_path: Path,
    key: list[tuple[str, int]],
    schema: TrajectorySchema,
    fields: list[dict[str, torch.Tensor]],
    metadata: list[TrajectoryMetadata],
    batch_size: torch.Size,
) -> None:
    """Write the trajectories as one tensor per field, published atomically for waiting ranks."""
    values, lengths = _pack(fields)
    cache = {
        "schema": schema.to_dict(),
        "metadata": [item.to_dict() for item in metadata],
        "batch_size": batch_size,
        "lengths": lengths,
        "values": values,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key_path = _key_path(cache_path)
    key_path.unlink(missing_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    # The key is published last, so a matching key always names a complete cache.
    temporary_key = key_path.with_name(f".{key_path.name}.{os.getpid()}.tmp")
    temporary_key.write_text(json.dumps([list(item) for item in key]))
    temporary_key.replace(key_path)
