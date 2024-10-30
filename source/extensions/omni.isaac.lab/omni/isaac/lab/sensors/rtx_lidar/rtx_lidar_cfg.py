# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
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
    data_types: list[str] = []
        # azimuth: torch.Tensor
        # beamId: torch.Tensor
        # data: torch.Tensor
        # distance: torch.Tensor
        # elevation: torch.Tensor
        # emitterId: torch.Tensor
        # index: torch.Tensor
        # intensity: torch.Tensor
        # materialId: torch.Tensor
        # normal: torch.Tensor
        # objectId: torch.Tensor
        # timestamp: torch.Tensor
        # velocity: torch.Tensor

    spawn: LidarCfg | None = MISSING