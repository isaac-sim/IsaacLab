# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optimization settings for AnymalC dynamics datasets."""

from isaaclab_neural import NeRDTrainingCfg

from isaaclab.utils import configclass


@configclass
class AnymalCNeRDTrainingCfg(NeRDTrainingCfg):
    """Optimizer and validation defaults independent of the model definition."""

    max_steps = 20000
    window = 32
    batch_size = 2048
    learning_rate = 0.002
    noise = 0.3
    horizons = (1, 10, 50)
