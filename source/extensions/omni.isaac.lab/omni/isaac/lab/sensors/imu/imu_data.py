# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass

import omni.isaac.lab.utils.math as math_utils


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

    lin_vel_w: torch.Tensor = None
    """Root angular velocity in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    ang_vel_w: torch.Tensor = None
    """Root angular velocity in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    lin_acc_w: torch.Tensor = None
    """Root linear acceleration in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    ang_acc_w: torch.Tensor = None
    """Root angular acceleration in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    @property
    def lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in body frame.

        Shape is (N, 3), where ``N`` is the number of environments.
        """
        return math_utils.quat_rotate_inverse(self.quat_w, self.lin_vel_w)
    
    @property
    def ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in body frame.

        Shape is (N, 3), where ``N`` is the number of environments.
        """
        return math_utils.quat_rotate_inverse(self.quat_w, self.ang_vel_w)
    
    @property
    def lin_acc_b(self) -> torch.Tensor:
        """Root linear acceleration in body frame.

        Shape is (N, 3), where ``N`` is the number of environments.
        """
        return math_utils.quat_rotate_inverse(self.quat_w, self.lin_acc_w)
    
    @property
    def ang_acc_b(self) -> torch.Tensor:
        """Root angular acceleration in body frame.

        Shape is (N, 3), where ``N`` is the number of environments.
        """
        return math_utils.quat_rotate_inverse(self.quat_w, self.ang_acc_w)
