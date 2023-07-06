# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass


@configclass
class SensorBaseCfg:
    """Configuration parameters for a sensor."""

    update_freq: float = 0.0
    """Update frequency of the sensor buffers (in Hz). Defaults to 0.0.

    If the sensor frequency is zero, then the sensor buffers are filled at every simulation step.
    """

    debug_vis: bool = False
    """Whether to visualize the sensor. Defaults to False."""
