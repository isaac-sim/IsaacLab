# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect accepted reset states through ordinary task callbacks."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

import numpy as np
import torch

from isaaclab.envs.reset import ResetStateBuffer

from .reset_state_collection_cfg import ResetStateCollectionCfg


def collect_reset_states(
    cfg: ResetStateCollectionCfg,
    env_ids: torch.Tensor,
    generate: Callable[[torch.Tensor], torch.Tensor | None]
    | Mapping[str, Callable[[torch.Tensor], torch.Tensor | None]],
    capture: Callable[[torch.Tensor], ResetStateBuffer.State],
    checks: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> tuple[ResetStateBuffer.State, torch.Tensor, dict[str, dict[str, int]]]:
    """Generate, check, and capture a complete set of reset states before training.

    Each generator receives all candidate environment indices on every attempt. Its optional
    boolean return mask reports failed construction, such as an unsuccessful IK solve. Named
    checks return additional acceptance masks. Only candidates passing every condition are
    captured. Named generators receive equal quotas, differing by at most one row.

    The callbacks own simulation synchronization and any settling; collection does not step
    simulation or advance episode bookkeeping itself. Torch, NumPy, and Python random states
    are restored after success or failure. Results are returned only after every quota is filled;
    callers can retain their previous states if collection fails.

    Args:
        cfg: Collection size, attempt limit, and seed.
        env_ids: Candidate environment indices, shape [N], on the simulation device.
        generate: A callback or non-empty mapping of generator names to callbacks. An optional
            returned boolean mask has shape [N]. The caller owns the mapping's group names.
        capture: Callback returning a tensor tree with one leading row per supplied index.
            Fields must keep the same structure, trailing shapes, dtypes, and devices across batches.
        checks: Named validity callbacks returning boolean tensors of shape [N]. The name
            ``generator`` is reserved for the generator's own validity mask.

    Returns:
        Captured tensor tree, generator indices for each stored row with shape [cfg.size], and
        attempted, accepted, and rejected counts by generator name. Generator indices follow
        the supplied mapping's iteration order; a single callback forms group zero.

    Raises:
        RuntimeError: A generator cannot fill its quota within ``cfg.max_batches``. The exception
            includes the failed collection's diagnostic counts.
        ValueError: Configuration, candidate environments, masks, or captured states are invalid.
    """
    if any(type(value) is not int or value < 1 for value in (cfg.size, cfg.max_batches)):
        raise ValueError("size and max_batches must be positive integers.")
    if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    generators = dict(generate) if isinstance(generate, Mapping) else {"default": generate}
    if env_ids.numel() == 0 or not generators or cfg.size < len(generators):
        raise ValueError("Collection needs candidate environments and at least one buffer row per generator.")
    checks = checks or {}
    if "generator" in checks:
        raise ValueError("The check name 'generator' is reserved for generator validity.")
    states: ResetStateBuffer.State = {}
    group_ids = torch.empty(cfg.size, dtype=torch.long, device=env_ids.device)
    collected = 0
    collection_report: dict[str, dict[str, int]] = {}
    devices = [env_ids.device.index] if env_ids.is_cuda else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(cfg.seed)
            if env_ids.is_cuda:
                with torch.cuda.device(env_ids.device):
                    torch.cuda.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            for group, (name, generator) in enumerate(generators.items()):
                quota = cfg.size // len(generators) + int(group < cfg.size % len(generators))
                accepted = 0
                report = collection_report[name] = {"attempted": 0, "accepted": 0}
                for _ in range(cfg.max_batches):
                    report["attempted"] += env_ids.numel()
                    reported = generator(env_ids)
                    masks = {"generator": reported} if reported is not None else {}
                    masks.update({key: check(env_ids) for key, check in checks.items()})
                    valid = torch.ones_like(env_ids, dtype=torch.bool)
                    for key, mask in masks.items():
                        if mask.shape != env_ids.shape or mask.dtype != torch.bool:
                            raise ValueError(f"{name}/{key} must return a boolean mask shaped {tuple(env_ids.shape)}.")
                        rejected = int((~mask).sum())
                        report[f"rejected/{key}"] = report.get(f"rejected/{key}", 0) + rejected
                        valid &= mask
                    selected = env_ids[valid][: quota - accepted]
                    if selected.numel():
                        if not _store_state(states, capture(selected), collected, selected.numel(), cfg.size):
                            raise ValueError("capture must return a non-empty tree of batched tensors.")
                        group_ids[collected : collected + selected.numel()] = group
                        collected += selected.numel()
                        accepted += selected.numel()
                        report["accepted"] = accepted
                    if accepted == quota:
                        break
                if accepted < quota:
                    raise RuntimeError(
                        f"Reset group '{name}' collected {accepted}/{quota} states after "
                        f"{cfg.max_batches} batches. Collection report: {collection_report}"
                    )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return states, group_ids, collection_report


def _store_state(
    destination: ResetStateBuffer.State,
    source: ResetStateBuffer.State,
    start: int,
    rows: int,
    size: int,
    path: str = "",
) -> int:
    if not isinstance(source, dict) or start and source.keys() != destination.keys():
        raise ValueError(f"Captured state structure changed at '{path}'.")
    leaves = 0
    for name, value in source.items():
        key = f"{path}/{name}"
        if isinstance(value, dict):
            if start and not isinstance(destination[name], dict):
                raise ValueError(f"Captured state structure changed at '{key}'.")
            if not start:
                destination[name] = {}
            leaves += _store_state(destination[name], value, start, rows, size, key)
        else:
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != rows:
                raise ValueError(f"Captured state '{key}' needs a leading dimension of {rows}.")
            if start:
                expected = destination[name]
                if not isinstance(expected, torch.Tensor) or (value.shape[1:], value.dtype, value.device) != (
                    expected.shape[1:],
                    expected.dtype,
                    expected.device,
                ):
                    raise ValueError(f"Captured tensor shape, dtype, or device changed at '{key}'.")
            else:
                destination[name] = value.new_empty((size, *value.shape[1:]))
            destination[name][start : start + rows].copy_(value.detach())
            leaves += 1
    return leaves
