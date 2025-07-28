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
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_kinematics_configuration import PinkKinematicsConfiguration
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
        if cfg.controlled_joint_names is None:
            raise ValueError("controlled_joint_names must be provided in the configuration")
        if cfg.all_joint_names is None:
            raise ValueError("all_joint_names must be provided in the configuration")

        self.cfg = cfg
        self.device = device

        # Initialize the Kinematics model used by pink IK to control robot
        self.pink_configuration = PinkKinematicsConfiguration(
            urdf_path=cfg.urdf_path,
            mesh_path=cfg.mesh_path,
            controlled_joint_names=cfg.controlled_joint_names,
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
        self.init_joint_positions = np.zeros(len(pink_joint_names))
        self.init_joint_positions[indices] = np.array(values)

        # Set the default targets for each task from the configuration
        for task in cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_joint_positions)
                continue
            task.set_target_from_configuration(self.pink_configuration)
        for task in cfg.fixed_input_tasks:
            task.set_target_from_configuration(self.pink_configuration)

        # Create joint ordering mappings
        self._setup_joint_ordering_mappings()

    def _setup_joint_ordering_mappings(self):
        """Setup joint ordering mappings between Isaac Lab and Pink conventions."""
        pink_joint_names = self.pink_configuration.all_joint_names_pinocchio_order
        isaac_lab_joint_names = self.cfg.all_joint_names
        
        # Type assertions for linter clarity
        assert pink_joint_names is not None, "pink_joint_names should not be None"
        assert isaac_lab_joint_names is not None, "isaac_lab_joint_names should not be None"

        # Create reordering arrays for all joints
        self.isaac_lab_to_pink_ordering = np.array(
            [isaac_lab_joint_names.index(pink_joint) for pink_joint in pink_joint_names]
        )
        self.pink_to_isaac_lab_ordering = np.array(
            [pink_joint_names.index(isaac_lab_joint) for isaac_lab_joint in isaac_lab_joint_names]
        )

        # Create reordering arrays for controlled joints only
        pink_controlled_joint_names = self.pink_configuration.controlled_joint_names_pinocchio_order
        isaac_lab_controlled_joint_names = self.cfg.controlled_joint_names
        
        # Type assertions for linter clarity
        assert pink_controlled_joint_names is not None, "pink_controlled_joint_names should not be None"
        assert isaac_lab_controlled_joint_names is not None, "isaac_lab_controlled_joint_names should not be None"

        self.isaac_lab_to_pink_controlled_ordering = np.array(
            [isaac_lab_controlled_joint_names.index(pink_joint) for pink_joint in pink_controlled_joint_names]
        )
        self.pink_to_isaac_lab_controlled_ordering = np.array(
            [pink_controlled_joint_names.index(isaac_lab_joint) for isaac_lab_joint in isaac_lab_controlled_joint_names]
        )


    """
    Operations.
    """

    def _reorder_array(self, input_array: np.ndarray, reordering_array: np.ndarray) -> np.ndarray:
        """Reorder the input array based on the provided ordering.

        Args:
            input_array: The array to reorder (numpy array).
            reordering_array: The indices to use for reordering (numpy array).

        Returns:
            Reordered array as a numpy array.
        """
        return input_array[reordering_array]

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
        if self.cfg.controlled_joint_indices is None:
            raise ValueError("controlled_joint_indices must be provided in the configuration")

        # Get the current controlled joint positions
        curr_controlled_joint_pos = [curr_joint_pos[i] for i in self.cfg.controlled_joint_indices]

        # Initialize joint positions for Pink, including the root and universal joints
        joint_positions_pink = self._reorder_array(curr_joint_pos, self.isaac_lab_to_pink_ordering)

        # Update Pink's robot configuration with the current joint positions
        self.pink_configuration.update(joint_positions_pink)

        # Solve IK using Pink's solver
        try:
            velocity = solve_ik(
                self.pink_configuration, 
                self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks, 
                dt, 
                solver="osqp",
                safety_break=self.cfg.fail_on_joint_limit_violation,
            )
            joint_angle_changes = velocity * dt
        except (AssertionError, Exception) as e:
            # Print warning and return the current joint positions as the target
            # Not using omni.log since its not available in CI during docs build
            if self.cfg.show_ik_warnings:
                print(
                    "Warning: IK quadratic solver could not find a solution! Did not update the target joint"
                    f" positions.\nError: {e}"
                )
            return torch.tensor(curr_controlled_joint_pos, device=self.device, dtype=torch.float32)

        # Reorder the joint angle changes back to Isaac Lab conventions
        joint_vel_isaac_lab = torch.tensor(
            self._reorder_array(joint_angle_changes, self.pink_to_isaac_lab_controlled_ordering),
            device=self.device,
            dtype=torch.float32,
        )

        # Add the velocity changes to the current joint positions to get the target joint positions
        target_joint_pos = torch.add(
            joint_vel_isaac_lab, 
            torch.tensor(curr_controlled_joint_pos, device=self.device, dtype=torch.float32)
        )

        return target_joint_pos
