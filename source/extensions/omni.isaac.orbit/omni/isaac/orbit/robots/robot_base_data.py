# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class RobotBaseData:
    """Data container for a robot."""

    ##
    # Frame states.
    ##

    root_state_w: torch.Tensor = None
    """Root state `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape is ``(count, 13)``."""

    ##
    # DOF states <- From simulation.
    ##

    dof_pos: torch.Tensor = None
    """DOF positions of all joints. Shape is ``(count, num_dof)``."""

    dof_vel: torch.Tensor = None
    """DOF velocities of all joints. Shape is ``(count, num_dof)``."""

    dof_acc: torch.Tensor = None
    """DOF acceleration of all joints. Shape is ``(count, num_dof)``."""

    ##
    # DOF commands -- Set into simulation.
    ##

    dof_pos_targets: torch.Tensor = None
    """DOF position targets provided to simulation. Shape is ``(count, num_dof)``.

    Note: The position targets are zero for explicit actuator models.
    """

    dof_vel_targets: torch.Tensor = None
    """DOF velocity targets provided to simulation. Shape is ``(count, num_dof)``.

    Note: The velocity targets are zero for explicit actuator models.
    """

    dof_effort_targets: torch.Tensor = None
    """DOF effort targets provided to simulation. Shape is ``(count, num_dof)``.

    Note: The torques are zero for implicit actuator models without feed-forward torque.
    """

    ##
    # DOF commands -- Explicit actuators.
    ##

    computed_torques: torch.Tensor = None
    """DOF torques computed from the actuator model (before clipping).
    Shape is ``(count, num_dof)``.

    Note: The torques are zero for implicit actuator models.
    """

    applied_torques: torch.Tensor = None
    """DOF torques applied from the actuator model (after clipping).
    Shape is ``(count, num_dof)``.

    Note: The torques are zero for implicit actuator models.
    """

    ##
    # Default actuator offsets <- From the actuator groups.
    ##

    actuator_pos_offset: torch.Tensor = None
    """Joint positions offsets applied by actuators when using "p_abs". Shape is ``(count, num_dof)``."""

    ##
    # Other Data.
    ##

    soft_dof_pos_limits: torch.Tensor = None
    """DOF positions limits for all joints. Shape is ``(count, num_dof, 2)``."""

    soft_dof_vel_limits: torch.Tensor = None
    """DOF velocity limits for all joints. Shape is ``(count, num_dof)``."""

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied DOF torques. Shape is ``(count, num_dof)``."""

    """
    Properties
    """

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is ``(count, 4)``."""
        return self.root_state_w[:, 3:7]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is ``(count, 3)``."""
        return self.root_state_w[:, 10:13]
