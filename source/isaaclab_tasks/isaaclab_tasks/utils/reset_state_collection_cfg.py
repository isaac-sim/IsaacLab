# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state collection limits and seed settings."""

from isaaclab.utils import configclass


@configclass
class ResetStateCollectionCfg:
    """Configuration for bounded reset-state collection."""

    size: int = 4096
    """Number of accepted states, shared approximately equally across named generators."""

    max_batches: int = 100
    """Maximum candidate batches per generator before collection fails."""

    seed: int = 0
    """Collection seed. The caller's Torch, NumPy, and Python random states are restored afterward."""
