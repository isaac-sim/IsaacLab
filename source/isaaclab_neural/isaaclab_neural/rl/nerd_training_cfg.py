# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optimization and validation settings for a recorded neural dynamics dataset."""

from isaaclab.utils import configclass


@configclass
class NeRDTrainingCfg:
    """Training settings; neither the model nor the solver owns this configuration."""

    max_steps: int = 20_000
    window: int = 16
    batch_size: int = 1024
    learning_rate: float = 2.0e-3
    noise: float = 0.1
    horizons: tuple[int, ...] = (1, 10)
