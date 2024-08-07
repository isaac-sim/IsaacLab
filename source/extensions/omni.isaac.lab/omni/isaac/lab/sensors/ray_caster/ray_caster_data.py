# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass, field


@dataclass
class RayCasterData:
    """Data container for the ray-cast sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame.

    Shape is (N, B, 3), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """

@dataclass
class RTXRayCasterInfo:
    numChannels: int
    numEchos: int
    numReturnsPerScan: int
    renderProductPath: str
    ticksPerScan: int
    transform: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    azimuth: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    beamId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    distance: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    elevation: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    emitterId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    index: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    intensity: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    materialId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    normal: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    objectId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    timestamp: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    velocity: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


def default_rtx_ray_caster_info():
    return RTXRayCasterInfo(
        numChannels=0,
        numEchos=0,
        numReturnsPerScan=0,
        renderProductPath='',
        ticksPerScan=0
    )


@dataclass
class RTXRayCasterData:
    azimuth: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    beamId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    data: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    distance: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    elevation: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    emitterId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    index: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    intensity: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    materialId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    normal: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    objectId: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    timestamp: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    velocity: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    info: RTXRayCasterInfo = field(default_factory=default_rtx_ray_caster_info)