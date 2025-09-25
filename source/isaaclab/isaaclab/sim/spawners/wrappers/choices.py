# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module provides utility functions to determine the spawning order of environments
when multiple assets are used in a multi-USD (Universal Scene Description) simulation setup.

These functions define how environment indices (`env_idx`) are mapped to asset indices
when spawning environments in Isaac Sim or similar simulators.

Each function in this file should follow a standardized signature:
    (current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int

Arguments:
- current_idx (int): The environment index (e.g., from 0 to n_tot - 1).
- total_prim_path (int): The total number of environments to spawn.
- num_assets (int): The number of different asset types available.
- **kwargs (dict): Keyword arguments.

Use this module when you want to control how environments are distributed among multiple assets.
"""
import random


def random_choice(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    """
    Randomly select an asset for the current index.

    Uses optional weights provided in `kwargs['weights']`. If no weights are given, all assets are equally likely.

    Each index is sampled independently according to optional weights.
    This means the overall distribution across all indices may not exactly match the desired weights.

    Use `deterministic_choice` if you want to maintain the exact proportion of each asset.

    Example weights:
        [1.0, 1.0, 1.0] -> uniform sampling rate for 3 assets
        [0.5, 0.3, 0.2] -> weighted sampling with asset 0 more likely
    """
    weights = kwargs.get("weights")
    if weights is None:
        weights = [1.0] * num_assets
    return random.choices(range(num_assets), weights=weights, k=1)[0]


def deterministic_choice(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    """
    Deterministically select assets to maintain the given weights as closely as possible.

    Example weights:
        [1.0, 1.0, 1.0] -> uniform sampling rate for 3 assets
        [0.5, 0.3, 0.2] -> weighted sampling with asset 0 more likely
    """
    weights = kwargs.get("weights")
    if weights is None:
        weights = [1.0] * num_assets

    total_weight = sum(weights)
    scaled_counts = [int(round(w * total_prim_path / total_weight)) for w in weights]
    seq = []

    for i, count in enumerate(scaled_counts):
        seq.extend([i] * count)

    if len(seq) < total_prim_path:
        seq.extend([seq[-1]] * (total_prim_path - len(seq)))
    elif len(seq) > total_prim_path:
        seq = seq[:total_prim_path]

    return seq[current_idx]


def sequential(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    """
    Assign assets in a simple round-robin fashion.

    Example sequence for 3 assets: 0, 1, 2, 0, 1, 2, ...

    """
    return current_idx % num_assets


def split(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    """
    Divide environments evenly among available assets.

    Example for 3 assets and 6 environments: 0, 0, 1, 1, 2, 2,
    """
    split_index = total_prim_path // num_assets
    return min(current_idx // split_index, num_assets - 1)
