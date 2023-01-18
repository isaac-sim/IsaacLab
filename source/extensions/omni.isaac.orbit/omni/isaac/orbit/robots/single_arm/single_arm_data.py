# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from ..robot_base_data import RobotBaseData


@dataclass
class SingleArmManipulatorData(RobotBaseData):
    """Data container for a robot arm with an optional gripper/tool."""

    ##
    # Frame states.
    ##

    ee_state_w: torch.Tensor = None
    """End-effector frame state `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape: (count, 13)."""

    ee_state_b: torch.Tensor = None
    """End-effector frame state `[pos, quat, lin_vel, ang_vel]` in base frame. Shape: (count, 13)."""

    tool_sites_state_w: torch.Tensor = None
    """Tool sites frames state `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape: (count, num_sites, 13)."""

    tool_sites_state_b: torch.Tensor = None
    """Tool sites frames state `[pos, quat, lin_vel, ang_vel]` in base frame. Shape: (count, num_sites, 13)."""

    ##
    # Dynamics state.
    ##

    ee_jacobian: torch.Tensor = None
    """Geometric Jacobian of the parent body of end-effector frame in simulation frame. Shape: (count, 6, arm_num_dof)."""

    mass_matrix: torch.Tensor = None
    """Mass matrix of the parent body of end-effector frame. Shape: (count, arm_num_dof, arm_num_dof)."""

    coriolis: torch.Tensor = None
    """Coriolis and centrifugal force on parent body of end-effector frame. Shape: (count, arm_num_dof)."""

    gravity: torch.Tensor = None
    """Generalized gravitational force on parent body of end-effector frame. Shape: (count, arm_num_dof)."""

    ##
    # DOF states.
    ##

    arm_dof_pos: torch.Tensor = None
    """Arm joint positions. Shape: (count, arm_num_dof)."""

    arm_dof_vel: torch.Tensor = None
    """Arm joint velocities. Shape: (count, arm_num_dof)."""

    arm_dof_acc: torch.Tensor = None
    """Arm joint acceleration. Shape: (count, arm_num_dof)."""

    tool_dof_pos: torch.Tensor = None
    """Tool joint positions. Shape: (count, tool_num_dof)."""

    tool_dof_vel: torch.Tensor = None
    """Tool joint velocities. Shape: (count, tool_num_dof)."""

    tool_dof_acc: torch.Tensor = None
    """Tool joint acceleration. Shape: (count, arm_num_dof)."""
