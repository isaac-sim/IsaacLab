# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World topology shared by every clone backend.

Asset definitions, world membership, and destination selections are the only stored facts.
Native names and placement belong to the backends that realize this topology.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cloner_strategies import sequential


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Asset prototypes and the world compositions instantiated from them."""

    asset_prototypes: tuple[Any, ...]
    """Asset prototype configurations. Repeated memberships refer to the same definition."""

    world_prototypes: np.ndarray
    """Flat asset-prototype indices. Repeated indices represent distinct instances."""

    world_prototype_starts: np.ndarray
    """Offsets into :attr:`world_prototypes`, starting with the shared world `-1`.

    Shared assets occupy ``world_prototypes[world_prototype_starts[0]:world_prototype_starts[1]]``.
    World prototype ``i`` occupies ``world_prototypes[world_prototype_starts[i + 1]:world_prototype_starts[i + 2]]``.
    An empty shared world starts with ``[0, 0]``.
    """

    destinations: np.ndarray
    """World-prototype index for each destination world; shared assets are not sampled."""


def make_clone_plan(
    asset_prototypes: Sequence[Any],
    world_prototypes: Sequence[Sequence[int]],
    num_worlds: int,
    *,
    weights: Sequence[float] | None = None,
    shared_assets: Sequence[int] = (),
    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
) -> ClonePlan:
    """Select world compositions without assigning names, transforms, or native resources.

    Args:
        asset_prototypes: Asset prototype definitions, retained by reference.
        world_prototypes: Asset indices in each world prototype, including repeated instances.
        num_worlds: Number of destination worlds.
        weights: Relative world-prototype weights; ``None`` gives every prototype equal weight.
        shared_assets: Asset indices instantiated once in the shared world ``-1``.
        clone_strategy: Function selecting world-prototype indices from weights.

    Returns:
        Flat topology with one leading shared-world slice and one selection per destination.
    """
    asset_prototypes = tuple(asset_prototypes)
    compositions = (tuple(shared_assets), *(tuple(world) for world in world_prototypes))
    if len(compositions) == 1:
        raise ValueError("At least one world prototype is required; an empty world is ().")
    members = np.asarray([asset for world in compositions for asset in world])
    if members.size and (
        not np.issubdtype(members.dtype, np.integer) or (members < 0).any() or (members >= len(asset_prototypes)).any()
    ):
        raise ValueError("World members must be integer indices into asset_prototypes.")
    weights = np.ones(len(compositions) - 1) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(compositions) - 1,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Each world prototype requires one finite, non-negative weight.")
    if weights.sum() <= 0 or num_worlds < 0:
        raise ValueError("Weights must have positive total mass and num_worlds must be non-negative.")
    destinations = np.asarray(clone_strategy(weights, num_worlds))
    if (
        destinations.shape != (num_worlds,)
        or not np.issubdtype(destinations.dtype, np.integer)
        or (destinations < 0).any()
        or (destinations >= len(weights)).any()
    ):
        raise ValueError("clone_strategy must select one valid world-prototype index per destination.")
    return ClonePlan(
        asset_prototypes=asset_prototypes,
        world_prototypes=members.astype(np.int32, copy=False),
        world_prototype_starts=np.cumsum([0, *(len(world) for world in compositions)], dtype=np.int64),
        destinations=destinations.astype(np.int32, copy=False),
    )


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """Create centered grid transforms as host arrays.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions [m].
        up_axis: Up axis for positions (``"z"``, ``"y"``, or ``"x"``).

    Returns:
        Positions [m], shape ``[N, 3]``, and identity xyzw orientations, shape ``[N, 4]``.
    """
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows, dtype=np.float32), np.arange(num_cols, dtype=np.float32), indexing="ij")
    ii = ii.reshape(-1)[:N]
    jj = jj.reshape(-1)[:N]
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    zero = np.zeros(N, dtype=np.float32)
    if up_axis.lower() == "z":
        positions = np.stack((x, y, zero), axis=1)
    elif up_axis.lower() == "y":
        positions = np.stack((x, zero, y), axis=1)
    else:
        positions = np.stack((zero, x, y), axis=1)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    return positions.astype(np.float32, copy=False), orientations
