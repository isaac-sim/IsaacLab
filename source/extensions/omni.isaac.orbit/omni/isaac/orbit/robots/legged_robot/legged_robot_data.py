# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from ..robot_base_data import RobotBaseData


@dataclass
class LeggedRobotData(RobotBaseData):
    """Data container for a legged robot."""

    ##
    # Frame states.
    ##

    root_vel_b: torch.Tensor = None
    """Root velocity `[lin_vel, ang_vel]` in base frame. Shape is ``(count, 6)``."""

    projected_gravity_b: torch.Tensor = None
    """Projection of the gravity direction on base frame. Shape is ``(count, 3)``."""

    feet_state_w: torch.Tensor = None
    """Feet sites frames state `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape is ``(count, num_feet, 13)``."""

    feet_pose_b: torch.Tensor = None
    """Feet frames pose `[pos, quat]` in base frame. Shape is ``(count, num_feet, 13)``."""

    """
    Properties
    """

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is ``(count, 3)``."""
        return self.root_vel_b[:, 0:3]

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_vel_b[:, 3:6]

    @property
    def feet_pos_w(self) -> torch.Tensor:
        """Feet position in simulation world frame. Shape is ``(count, num_feet, 3)``."""
        return self.feet_state_w[..., :3]

    @property
    def feet_quat_w(self) -> torch.Tensor:
        """Feet orientation (w, x, y, z) in simulation world frame. Shape is ``(count, num_feet, 4)``."""
        return self.feet_state_w[:, 3:7]

    @property
    def feet_lin_vel_w(self) -> torch.Tensor:
        """Feet linear velocity in simulation world frame. Shape is ``(count, num_feet, 3)``."""
        return self.feet_state_w[:, 7:10]

    @property
    def feet_ang_vel_w(self) -> torch.Tensor:
        """Feet angular velocity in simulation world frame. Shape is ``(count, num_feet, 3)``."""
        return self.feet_state_w[:, 10:13]

    @property
    def feet_pos_b(self) -> torch.Tensor:
        """Feet position in base frame. Shape is ``(count, num_feet, 3)``."""
        return self.feet_pose_b[..., :3]

    @property
    def feet_quat_b(self) -> torch.Tensor:
        """Feet orientation (w, x, y, z) in base world frame. Shape is ``(count, num_feet, 4)``."""
        return self.feet_pose_b[:, 3:7]
