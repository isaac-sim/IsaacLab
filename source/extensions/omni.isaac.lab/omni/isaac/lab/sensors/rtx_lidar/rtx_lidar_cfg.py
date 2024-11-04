# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.sensors import SensorBaseCfg
from omni.isaac.lab.sim import LidarCfg
from omni.isaac.lab.utils import configclass

from .rtx_lidar import RtxLidar


@configclass
class RtxLidarCfg(SensorBaseCfg):

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type = RtxLidar
    offset: OffsetCfg = OffsetCfg()
    optional_data_types: list[
        Literal["azimuth", "beamId", "elevation", "emitterId", "index", "materialId", "normal", "objectId", "velocity"]
    ] = []
    data_frame: Literal["world", "sensor"] = "world"
    spawn: LidarCfg = MISSING
