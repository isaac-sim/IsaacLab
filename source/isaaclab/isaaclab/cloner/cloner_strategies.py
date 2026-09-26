# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def random(combinations: np.ndarray, num_clones: int) -> np.ndarray:
    """Randomly assign prototypes to environments.

    Each environment is assigned a random prototype combination sampled uniformly from
    :attr:`combinations`.

    Args:
        combinations: Array of shape (num_combos, num_prototypes) containing all possible
            prototype combinations.
        num_clones: Number of environments to assign combinations to.

    Returns:
        Array of shape (num_clones, num_prototypes) containing the chosen prototype
        combination for each environment.
    """
    return combinations[np.random.randint(len(combinations), size=num_clones)]


def sequential(combinations: np.ndarray, num_clones: int) -> np.ndarray:
    """Group :func:`round_robin` assignments by first occurrence, preserving combination counts.

    Args:
        combinations: Prototype combinations, shape (num_combos, num_prototypes). Duplicates add weight.
        num_clones: Number of environments.

    Returns:
        Chosen combinations, shape (num_clones, num_prototypes), with the input dtype.
    """
    if num_clones == 0:
        return combinations[:0].copy()
    _, first, inverse, counts = np.unique(
        combinations, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    cycles, remainder = divmod(num_clones, len(combinations))
    counts = cycles * counts + np.bincount(inverse[:remainder], minlength=len(counts))
    order = np.argsort(first)
    return np.repeat(combinations[first[order]], counts[order], axis=0)


def round_robin(combinations: np.ndarray, num_clones: int) -> np.ndarray:
    """Deterministically assign prototypes to environments in round-robin fashion.

    Each environment is assigned a prototype combination based on its index modulo the
    number of available combinations.

    Args:
        combinations: Array of shape (num_combos, num_prototypes) containing all possible
            prototype combinations.
        num_clones: Number of environments to assign combinations to.

    Returns:
        Array of shape (num_clones, num_prototypes) containing the chosen prototype
        combination for each environment.
    """
    return combinations[np.arange(num_clones) % len(combinations)]
