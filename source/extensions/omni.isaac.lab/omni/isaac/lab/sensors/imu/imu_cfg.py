# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import RED_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .imu import IMU


@configclass
class IMUCfg(SensorBaseCfg):
    """Configuration for a camera sensor."""

    class_type: type = IMU

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    offset: OffsetCfg = OffsetCfg()

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    """The configuration object for the visualization markers. Defaults to RED_ARROW_X_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
