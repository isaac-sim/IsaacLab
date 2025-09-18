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
    (current_idx: int, total_prim_path: int, num_assets: int) -> int

Arguments:
- current_idx (int): The environment index (e.g., from 0 to n_tot - 1).
- total_prim_path (int): The total number of environments to spawn.
- num_assets (int): The number of different asset types available.

Use this module when you want to control how environments are distributed among multiple assets.
"""
import random


def random_choice(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    weights = kwargs.get("weights", None)
    if weights is None:
        weights = [1.0] * num_assets
    return random.choices(range(num_assets), weights=weights, k=1)[0]


def deterministic_choice(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    weights = kwargs.get("weights", None)
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
    return current_idx % num_assets


def split(current_idx: int, total_prim_path: int, num_assets: int, **kwargs) -> int:
    split_index = total_prim_path // num_assets
    return min(current_idx // split_index, num_assets - 1)
