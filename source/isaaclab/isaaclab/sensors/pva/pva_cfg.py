# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .pva import Pva


@configclass
class PvaCfg(SensorBaseCfg):
    """Configuration for a Pose Velocity Acceleration (PVA) sensor."""

    class_type: type[Pva] | str = "{DIR}.pva:Pva"

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame [m]. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    """The configuration object for the visualization markers. Defaults to RED_ARROW_X_MARKER_CFG.

    This attribute is only used when debug visualization is enabled.
    """
