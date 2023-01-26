# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class ArticulatedObjectData:
    """Data container for an articulated object."""

    ##
    # Frame states.
    ##

    root_state_w: torch.Tensor = None
    """
    Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
    Shape is ``(count, 13)``.
    """

    sites_state_w: torch.Tensor = None
    """
    Sites frames state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
    Shape is ``(count, num_sites, 13)``.
    """

    sites_state_b: torch.Tensor = None
    """
    Sites frames state ``[pos, quat, lin_vel, ang_vel]`` in base frame.
    Shape: (count, num_sites, 13).
    """

    ##
    # DOF states <- From simulation.
    ##

    dof_pos: torch.Tensor = None
    """DOF positions of all joints. Shape is ``(count, num_dof)``."""

    dof_vel: torch.Tensor = None
    """DOF velocities of all joints. Shape is ``(count, num_dof)``."""

    dof_acc: torch.Tensor = None
    """DOF acceleration of all joints. Shape is ``(count, num_dof)``."""

    """
    Properties
    """

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation in quaternion (w, x, y, z) in simulation world frame. Shape is ``(count, 4)``."""
        return self.root_state_w[:, 3:7]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_state_w[:, 10:13]
