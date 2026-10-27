# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


def normalize_weights(weights: list[float]) -> list[float]:
    """Scale non-negative weights to sum to one without modifying the input.

    Args:
        weights: Finite, non-negative weights with a positive total.

    Returns:
        Normalized weights in the input order.

    Raises:
        ValueError: If a weight is negative or the total is not positive.
    """
    total = sum(weights)
    if total <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("Weights must be non-negative with a positive total.")
    return [weight / total for weight in weights]
