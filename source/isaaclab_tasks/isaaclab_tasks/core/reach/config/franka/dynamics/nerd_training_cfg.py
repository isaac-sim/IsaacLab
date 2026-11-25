# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optimization settings for Franka dynamics datasets."""

from isaaclab_neural import NeRDTrainingCfg

from isaaclab.utils import configclass


@configclass
class FrankaNeRDTrainingCfg(NeRDTrainingCfg):
    """Optimizer and validation defaults independent of the model definition."""

    max_steps = 24000
    window = 16
    batch_size = 1024
    learning_rate = 0.002
    noise = 0.1
    horizons = (1, 30)
