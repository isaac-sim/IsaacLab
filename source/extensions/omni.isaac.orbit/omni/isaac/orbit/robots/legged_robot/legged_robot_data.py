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

    feet_state_b: torch.Tensor = None
    """Feet frames state `[pos, quat, lin_vel, ang_vel]` in base frame. Shape is ``(count, num_feet, 13)``."""

    feet_air_time: torch.Tensor = None
    """Time spent (in s) during swing phase of each leg since last contact. Shape is ``(count, num_feet)``."""

    ##
    # Proprioceptive sensors.
    ##

    feet_contact_forces: torch.Tensor = None
    """Feet contact wrenches `[force, torque]` in simulation world frame. Shape is ``(count, num_feet, 6)``."""

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
