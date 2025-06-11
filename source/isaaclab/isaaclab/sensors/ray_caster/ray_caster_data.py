# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


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
