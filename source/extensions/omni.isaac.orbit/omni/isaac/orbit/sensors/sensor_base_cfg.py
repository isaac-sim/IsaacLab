# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import ClassVar, Optional

from omni.isaac.orbit.utils import configclass


@configclass
class SensorBaseCfg:
    """Configuration parameters for a sensor."""

    cls_name: ClassVar[str] = MISSING
    """Name of the associated sensor class."""

    prim_path_expr: Optional[str] = None
    """Relative path to the prim on which the sensor should be attached. Defaults to None."""

    update_period: float = 0.0
    """Update period of the sensor buffers (in seconds). Defaults to 0.0 (update every step)."""

    history_length: int = 0
    """Number of past frames to store in the sensor buffers. Defaults to 0 (no history)."""

    debug_vis: bool = False
    """Whether to visualize the sensor. Defaults to False."""
