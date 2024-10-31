# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

RTX_LIDAR_INFO_FIELDS = {"numChannels": int, 
                         "numEchos": int, 
                         "numReturnsPerScan": int, 
                         "renderProductPath": str,
                         "ticksPerScan": int, 
                         "transform": torch.Tensor }

@dataclass
class RtxLidarData:

    info: list[dict[str, Any]] = None    

    output: TensorDict = None
        # Always returned
            # data: torch.Tensor
            # distance: torch.Tensor
            # intensity: torch.Tensor
        # Optional
            # azimuth: torch.Tensor
            # beamId: torch.Tensor
            # elevation: torch.Tensor
            # emitterId: torch.Tensor
            # index: torch.Tensor
            # materialId: torch.Tensor
            # normal: torch.Tensor
            # objectId: torch.Tensor
            # timestamp: torch.Tensor
            # velocity: torch.Tensor

    