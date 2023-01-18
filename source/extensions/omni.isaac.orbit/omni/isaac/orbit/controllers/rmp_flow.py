# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import Tuple

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula import RmpFlow

from omni.isaac.orbit.utils import configclass


@configclass
class RmpFlowControllerCfg:
    """Configuration for RMP-Flow controller (provided through LULA library)."""

    config_file: str = MISSING
    """Path to the configuration file for the controller."""
    urdf_file: str = MISSING
    """Path to the URDF model of the robot."""
    collision_file: str = MISSING
    """Path to collision model description of the robot."""
    frame_name: str = MISSING
    """Name of the robot frame for task space (must be present in the URDF)."""
    evaluations_per_frame: int = MISSING
    """Number of substeps during Euler integration inside LULA world model."""
    ignore_robot_state_updates: bool = False
    """If true, then state of the world model inside controller is rolled out. (default: False)."""


class RmpFlowController:
    """Wraps around RMP-Flow from IsaacSim for batched environments."""

    def __init__(self, cfg: RmpFlowControllerCfg, prim_paths_expr: str, device: str):
        # store input
        self.cfg = cfg
        self._device = device

        print(f"[INFO]: Loading controller URDF from: {self.cfg.urdf_file}")
        # obtain the simulation time
        physics_dt = SimulationContext.instance().get_physics_dt()
        # find all prims
        self._prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
        self.num_robots = len(self._prim_paths)
        # create all franka robots references and their controllers
        self.articulation_policies = list()
        for prim_path in self._prim_paths:
            # add robot reference
            robot = Articulation(prim_path)
            robot.initialize()
            # add controller
            rmpflow = RmpFlow(
                rmpflow_config_path=self.cfg.config_file,
                urdf_path=self.cfg.urdf_file,
                robot_description_path=self.cfg.collision_file,
                end_effector_frame_name=self.cfg.frame_name,
                evaluations_per_frame=self.cfg.evaluations_per_frame,
                ignore_robot_state_updates=self.cfg.ignore_robot_state_updates,
            )
            # wrap rmpflow to connect to the Franka robot articulation
            articulation_policy = ArticulationMotionPolicy(robot, rmpflow, physics_dt)
            self.articulation_policies.append(articulation_policy)
        # get number of active joints
        self.active_dof_names = self.articulation_policies[0].get_motion_policy().get_active_joints()
        self.num_dof = len(self.active_dof_names)
        # create buffers
        # -- for storing command
        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device)
        # -- for policy output
        self.dof_pos_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)
        self.dof_vel_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        return 7

    """
    Operations.
    """

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        pass

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        # store command
        self._command[:] = command

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target joint positions and velocity commands.
        """
        # convert command to numpy
        command = self._command.cpu().numpy()
        # compute control actions
        for i, policy in enumerate(self.articulation_policies):
            # enable type-hinting
            policy: ArticulationMotionPolicy
            # set rmpflow target to be the current position of the target cube.
            policy.get_motion_policy().set_end_effector_target(
                target_position=command[i, 0:3], target_orientation=command[i, 3:7]
            )
            # apply action on the robot
            action = policy.get_next_articulation_action()
            # copy actions into buffer
            # TODO: Make this more efficient?
            for dof_index in range(self.num_dof):
                self.dof_pos_target[i, dof_index] = action.joint_positions[dof_index]
                self.dof_vel_target[i, dof_index] = action.joint_velocities[dof_index]

        return self.dof_pos_target, self.dof_vel_target
