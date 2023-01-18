# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from ..legged_robot import LeggedRobotData
from ..single_arm import SingleArmManipulatorData

__all__ = ["MobileManipulatorData", "LeggedMobileManipulatorData"]


@dataclass
class MobileManipulatorData(SingleArmManipulatorData):
    """Data container for a mobile manipulator with an optional gripper/tool."""

    ##
    # Frame states.
    ##

    root_vel_b: torch.Tensor = None
    """Root velocity `[lin_vel, ang_vel]` in base frame. Shape is ``(count, 6)``."""

    projected_gravity_b: torch.Tensor = None
    """Projection of the gravity direction on base frame. Shape is ``(count, 3)``."""

    ##
    # DOF states.
    ##

    base_dof_pos: torch.Tensor = None
    """Base positions. Shape is ``(count, base_num_dof)``."""

    base_dof_vel: torch.Tensor = None
    """Base velocities. Shape is ``(count, base_num_dof)``."""

    base_dof_acc: torch.Tensor = None
    """Base acceleration. Shape is ``(count, base_num_dof)``."""

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


@dataclass
class LeggedMobileManipulatorData(MobileManipulatorData, LeggedRobotData):
    """Data container for a legged mobile manipulator with an optional gripper/tool."""

    pass
