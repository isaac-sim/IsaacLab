# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for visualization markers."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class VisualizationMarkersCfg:
    """A class to configure a :class:`VisualizationMarkers`."""

    prim_path: str = MISSING
    """The prim path where the :class:`UsdGeom.PointInstancer` will be created."""

    markers: dict[str, SpawnerCfg] = MISSING
    """The dictionary of marker configurations.

    The key is the name of the marker, and the value is the configuration of the marker.
    The key is used to identify the marker in the class.
    """
