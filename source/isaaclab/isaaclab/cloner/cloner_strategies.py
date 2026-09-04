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
