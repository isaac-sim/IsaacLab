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
    """Configuration for the RtxLidar sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type = RtxLidar

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.

    Note:
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """
    optional_data_types: list[
        Literal["azimuth", "beamId", "elevation", "emitterId", "index", "materialId", "normal", "objectId", "velocity"]
    ] = []
    """The optional output data types to include in RtxLidarData.

    Please refer to the :class:'RtxLidar' and :class:'RtxLidarData' for a list and description of available data types.
    """
    data_frame: Literal["world", "sensor"] = "world"
    """The frame to represent the output.data.

    If 'world' the output.data will be in the world frame. If 'sensor' the output.data will be in the sensor frame."""
    spawn: LidarCfg = MISSING
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """
