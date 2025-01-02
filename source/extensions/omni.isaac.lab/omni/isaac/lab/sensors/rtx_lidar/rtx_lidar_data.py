# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

RTX_LIDAR_INFO_FIELDS = {
    "numChannels": int,
    "numEchos": int,
    "numReturnsPerScan": int,
    "renderProductPath": str,
    "ticksPerScan": int,
    "transform": torch.Tensor,
}


@dataclass
class RtxLidarData:
    """Data container for the RtxLidar sensor."""

    info: list[dict[str, Any]] = None
    """The static descriptive information from the lidar sensor.

    Each dictionary corresponds to an instance of a lidar and contains the following fields from the
    "RtxSensorCpuIsaacCreateRTXLidarScanBuffer" annotator:
        numChannels: The maximum number of channels.
        numEchos: The maximum number echos.
        numReturnsPerScan: The maximum number of returns possible in the output.
        renderProductPath: The render product from the camera prim sensor.
        ticksPerScan: The maximum number of ticks.
        transform: Transform of the latest data added to the scan buffer for transforming the data to world space.

    The product  of ticksPerScan, numChannels, and numEchos will be the same as the number of returns if you initialize
    the annotator with annotator.initialize(keepOnlyPositiveDistance=False) before attaching the render product.
    """
    output: dict[str, torch.Tensor] = None
    """The data that changes every sample. Some fields of the out will always be returned and some are optionally
    returned when configured in RtxLidarCfg.optional_data_types.

    The following keys will ALWAYS be returned:
        data: The position [x,y,z] of each return in meter expressed in the RtxLidarCfg.data_frame. If 'world' the
            data will be returned relative to simulation world. If 'sensor' the data will be returned relative to
            sensor frame.
        distance: The distance of the return hit from sensor origin in meters.
        intensity: The intensity value in the range [0.0, 1.0] of each return.
        timestamp: The time since sensor creation time in nanoseconds for each return.

    The following keys will OPTIONALLY be returned:
        azimuth: The horizontal polar angle (radians) of the return.
        elevation: the vertical polar angle (radians) of the return.
        normal: The normal at the hit location in world coordinates.
        velocity: The normalized velocity at the hit location in world coordinates.
        emmitterId: Same as the channel unless the RTX Lidar Config Parameters is set up so emitters fire through
            different channels.
        materialId: The sensor material Id at the hit location. Same as index from rtSensorNameToIdMap setting in the RTX Sensor Visual Materials.
        objectId: The object Id at the hit location. The objectId can be used to get the prim path of the object with the following code:
            from omni.syntheticdata._syntheticdata import acquire_syntheticdata_interface
            primpath = acquire_syntheticdata_interface().get_uri_from_instance_segmentation_id(object_id)
    """
