# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK controller implementation for IsaacLab.

This module provides integration between Pink inverse kinematics solver and IsaacLab.
Pink is a differentiable inverse kinematics solver framework that provides task-space control capabilities.
"""

import numpy as np
import torch

from pink import solve_ik
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pinocchio.robot_wrapper import RobotWrapper

from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.null_space_posture_task import NullSpacePostureTask
from isaaclab.utils.string import resolve_matching_names_values

from .pink_ik_cfg import PinkIKControllerCfg


class PinkIKController:
    """Integration of Pink IK controller with Isaac Lab.

    The Pink IK controller is available at: https://github.com/stephane-caron/pink
    """

    def __init__(self, cfg: PinkIKControllerCfg, robot_cfg: ArticulationCfg, device: str):
        """Initialize the Pink IK Controller.

        Args:
            cfg: The configuration for the controller.
            device: The device to use for computations (e.g., 'cuda:0').
        """
        # Initialize the robot model from URDF and mesh files
        self.robot_wrapper = RobotWrapper.BuildFromURDF(cfg.urdf_path, cfg.mesh_path, root_joint=None)
        self.pink_configuration = Configuration(
            self.robot_wrapper.model, self.robot_wrapper.data, self.robot_wrapper.q0
        )

        # Find the initial joint positions by matching Pink's joint names to robot_cfg.init_state.joint_pos,
        # where the joint_pos keys may be regex patterns and the values are the initial positions.
        # We want to assign to each Pink joint name the value from the first matching regex key in joint_pos.
        pink_joint_names = self.pink_configuration.model.names.tolist()[1:]
        joint_pos_dict = robot_cfg.init_state.joint_pos

        # Use resolve_matching_names_values to match Pink joint names to joint_pos values
        indices, names, values = resolve_matching_names_values(
            joint_pos_dict, pink_joint_names, preserve_order=False, require_all_keys_matched=False
        )
        if len(indices) != len(pink_joint_names):
            unmatched = [name for name in pink_joint_names if name not in names]
            raise KeyError(
                "Could not find a match for all Pink joint names in robot_cfg.init_state.joint_pos. "
                f"Unmatched: {unmatched}, Expected: {pink_joint_names}"
            )
        self.init_joint_positions = np.array(values)

        # Set the default targets for each task from the configuration
        for task in cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_joint_positions)
                continue
            task.set_target_from_configuration(self.pink_configuration)
        for task in cfg.fixed_input_tasks:
            task.set_target_from_configuration(self.pink_configuration)

        # Map joint names from Isaac Lab to Pink's joint conventions
        self.pink_joint_names = self.robot_wrapper.model.names.tolist()[1:]  # Skip the root and universal joints
        self.isaac_lab_joint_names = cfg.joint_names

        # Frame task link names
        self.frame_task_link_names = []
        for task in cfg.variable_input_tasks:
            if isinstance(task, FrameTask):
                self.frame_task_link_names.append(task.frame)

        # Create reordering arrays for joint indices
        self.isaac_lab_to_pink_ordering = [
            self.isaac_lab_joint_names.index(pink_joint) for pink_joint in self.pink_joint_names
        ]
        self.pink_to_isaac_lab_ordering = [
            self.pink_joint_names.index(isaac_lab_joint) for isaac_lab_joint in self.isaac_lab_joint_names
        ]

        self.cfg = cfg
        self.device = device

    def reorder_array(self, input_array: list[float], reordering_array: list[int]) -> list[float]:
        """Reorder the input array based on the provided ordering.

        Args:
            input_array: The array to reorder.
            reordering_array: The indices to use for reordering.

        Returns:
            Reordered array.
        """
        return [input_array[i] for i in reordering_array]

    def initialize(self):
        """Initialize the internals of the controller.

        This method is called during setup but before the first compute call.
        """
        pass

    def update_null_space_joint_targets(self, curr_joint_pos: np.ndarray):
        """Update the null space joint targets.

        Args:
            curr_joint_pos: The current joint positions.
        """
        for task in self.cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(curr_joint_pos)

    def compute(
        self,
        curr_joint_pos: np.ndarray,
        dt: float,
    ) -> torch.Tensor:
        """Compute the target joint positions based on current state and tasks.

        Args:
            curr_joint_pos: The current joint positions.
            dt: The time step for computing joint position changes.

        Returns:
            The target joint positions as a tensor.
        """
        # Initialize joint positions for Pink, including the root and universal joints
        joint_positions_pink = np.array(self.reorder_array(curr_joint_pos, self.isaac_lab_to_pink_ordering))

        # Update Pink's robot configuration with the current joint positions
        self.pink_configuration.update(joint_positions_pink)

        # pink.solve_ik can raise an exception if the solver fails
        try:
            velocity = solve_ik(
                self.pink_configuration,
                self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks,
                dt,
                solver="osqp",
                safety_break=self.cfg.fail_on_joint_limit_violation,
            )
            Delta_q = velocity * dt
        except (AssertionError, Exception) as e:
            # Print warning and return the current joint positions as the target
            # Not using omni.log since its not available in CI during docs build
            if self.cfg.show_ik_warnings:
                print(
                    "Warning: IK quadratic solver could not find a solution! Did not update the target joint"
                    f" positions.\nError: {e}"
                )
            return torch.tensor(curr_joint_pos, device=self.device, dtype=torch.float32)

        # Discard the first 6 values (for root and universal joints)
        pink_joint_angle_changes = Delta_q

        # Reorder the joint angle changes back to Isaac Lab conventions
        joint_vel_isaac_lab = torch.tensor(
            self.reorder_array(pink_joint_angle_changes, self.pink_to_isaac_lab_ordering),
            device=self.device,
            dtype=torch.float,
        )

        # Add the velocity changes to the current joint positions to get the target joint positions
        target_joint_pos = torch.add(
            joint_vel_isaac_lab, torch.tensor(curr_joint_pos, device=self.device, dtype=torch.float32)
        )

        return target_joint_pos
