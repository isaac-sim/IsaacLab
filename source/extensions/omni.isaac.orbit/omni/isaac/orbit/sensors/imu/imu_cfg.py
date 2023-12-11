# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.markers import VisualizationMarkersCfg
from omni.isaac.orbit.markers.config import RED_ARROW_X_MARKER_CFG

from ..sensor_base_cfg import SensorBaseCfg
from .imu import IMU


@configclass
class IMUCfg(SensorBaseCfg):
    """Configuration for a camera sensor."""

    class_type: IMU = IMU

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    """The configuration object for the visualization markers. Defaults to RED_ARROW_X_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
