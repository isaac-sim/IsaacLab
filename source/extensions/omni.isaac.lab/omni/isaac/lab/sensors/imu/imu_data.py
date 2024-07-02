# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class IMUData:
    """Data container for the IMU sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where ``N`` is the number of sensors.
    """

    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame.

    Shape is (N, 4), where ``N`` is the number of sensors.
    """

    ang_vel_b: torch.Tensor = None
    """Root angular velocity in body frame.

    Shape is (N, B, 3), where ``N`` is the number of sensors and ``B`` is the number of bodies in each sensor.
    """

    lin_acc_b: torch.Tensor = None
    """Root linear acceleration in body frame.

    Shape is (N, B, 3), where ``N`` is the number of sensors and ``B`` is the number of bodies in each sensor.
    """
