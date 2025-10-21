# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ImuData:
    """Data container for the Imu sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame.

    Shape is (N, 4), where ``N`` is the number of environments.
    """

    projected_gravity_b: torch.Tensor = None
    """Gravity direction unit vector projected on the imu frame.

    Shape is (N,3), where ``N`` is the number of environments.
    """

    lin_vel_b: torch.Tensor = None
    """IMU frame angular velocity relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    ang_vel_b: torch.Tensor = None
    """IMU frame angular velocity relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    lin_acc_b: torch.Tensor = None
    """IMU frame linear acceleration relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    ang_acc_b: torch.Tensor = None
    """IMU frame angular acceleration relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """
