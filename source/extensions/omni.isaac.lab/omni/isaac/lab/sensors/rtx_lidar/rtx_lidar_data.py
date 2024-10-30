# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

@dataclass
class RtxLidarInfo:
    numChannels: int = 0
    numEchos: int = 0
    numReturnsPerScan: int = 0
    renderProductPath: str = ''
    ticksPerScan: int = 0
    transform: bool = False

@dataclass
class RtxLidarData:

    info: list[dict[str, Any]] = None    
        # numChannels: int = 0
        # numEchos: int = 0
        # numReturnsPerScan: int = 0
        # renderProductPath: str = ''
        # ticksPerScan: int = 0
        # transform: bool = False

    output: TensorDict = None
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

    