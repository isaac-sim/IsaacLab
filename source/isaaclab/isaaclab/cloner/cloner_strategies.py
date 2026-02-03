# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def random(combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor:
    """Randomly assign prototypes to environments.

    Each environment is assigned a random prototype combination sampled uniformly from
    :attr:`combinations`.

    Args:
        combinations: Tensor of shape (num_combos, num_prototypes) containing all possible
            prototype combinations.
        num_clones: Number of environments to assign combinations to.
        device: Torch device on which the output tensor is allocated.

    Returns:
        Tensor of shape (num_clones, num_prototypes) containing the chosen prototype
        combination for each environment.
    """
    chosen = combinations[torch.randint(len(combinations), (num_clones,), device=device)]
    return chosen


def sequential(combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor:
    """Deterministically assign prototypes to environments in round-robin fashion.

    Each environment is assigned a prototype combination based on its index modulo the
    number of available combinations.

    Args:
        combinations: Tensor of shape (num_combos, num_prototypes) containing all possible
            prototype combinations.
        num_clones: Number of environments to assign combinations to.
        device: Torch device on which the output tensor is allocated.

    Returns:
        Tensor of shape (num_clones, num_prototypes) containing the chosen prototype
        combination for each environment.
    """
    chosen = combinations[torch.arange(num_clones, device=device) % len(combinations)]
    return chosen
