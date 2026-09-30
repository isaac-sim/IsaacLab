# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def random(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Randomly assign world prototypes according to their weights.

    Args:
        weights: Relative sampling weights, one per world prototype.
        num_clones: Number of environments to assign combinations to.

    Returns:
        Integer array of shape [num_clones] selecting a world prototype per environment.
    """
    return np.random.choice(len(weights), size=num_clones, p=weights / weights.sum())


def sequential(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Assign contiguous world groups using evenly spaced samples of the weighted distribution.

    Args:
        weights: Relative sampling weights, one per world prototype.
        num_clones: Number of environments to assign combinations to.

    Returns:
        Integer array of shape [num_clones] selecting a world prototype per environment.
    """
    samples = (np.arange(num_clones) + 0.5) / max(num_clones, 1)
    return np.searchsorted(np.cumsum(weights) / weights.sum(), samples, side="right")
