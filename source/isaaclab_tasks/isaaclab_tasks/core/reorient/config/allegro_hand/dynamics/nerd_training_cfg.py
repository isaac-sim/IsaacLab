# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optimization settings for AllegroHand dynamics datasets."""

from isaaclab_neural import NeRDTrainingCfg

from isaaclab.utils import configclass


@configclass
class AllegroHandNeRDTrainingCfg(NeRDTrainingCfg):
    """Optimizer and validation defaults independent of the model definition."""

    max_steps = 60000
    window = 0
    batch_size = 8192
    learning_rate = 0.001
    noise = 0.0
    horizons = ()
