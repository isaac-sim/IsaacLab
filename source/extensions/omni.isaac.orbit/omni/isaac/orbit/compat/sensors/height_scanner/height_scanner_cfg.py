# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the height scanner sensor."""

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass


@configclass
class HeightScannerCfg:
    """Configuration for the height-scanner sensor."""

    sensor_tick: float = 0.0
    """Simulation seconds between sensor buffers. Defaults to 0.0."""
    points: list = MISSING
    """The 2D scan points to query ray-casting from. Results are reported in this order."""
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The offset from the frame the sensor is attached to. Defaults to (0.0, 0.0, 0.0)."""
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Unit direction for the scanner ray-casting. Defaults to (0.0, 0.0, -1.0)."""
    max_distance: float = 100.0
    """Maximum distance from the sensor to ray cast to. Defaults to 100.0."""
    filter_prims: list[str] = list()
    """A list of prim names to ignore ray-cast collisions with. Defaults to empty list."""
