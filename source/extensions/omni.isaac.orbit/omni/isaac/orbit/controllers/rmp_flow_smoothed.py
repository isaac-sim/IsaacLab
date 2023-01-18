# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from typing import Tuple

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.prims import find_matching_prim_paths
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula import RmpFlow


class RmpFlowController:
    """Wraps around RMP-Flow from IsaacSim for batched environments."""

    def __init__(self, prim_paths_expr: str, robot_name: str, dt: float, device: str):
        # store input
        self._device = device
        # Load configuration for the controller
        # Note: RMP-Flow config files for supported robots are stored in the motion_generation extension
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        # configuration for the robot
        if robot_name == "franka":
            rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs", "franka")
            robot_description_path = os.path.join(rmp_config_dir, "rmpflow", "robot_descriptor.yaml")
            robot_urdf_path = os.path.join(rmp_config_dir, "lula_franka_gen.urdf")
            rmpflow_config_path = os.path.join(rmp_config_dir, "rmpflow", "franka_rmpflow_common.yaml")
            self.arm_dof_num = 7
            self.ee_frame_name = "right_gripper"
        elif robot_name == "ur10":
            rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs", "ur10")
            robot_description_path = os.path.join(rmp_config_dir, "rmpflow", "ur10_robot_description.yaml")
            robot_urdf_path = os.path.join(rmp_config_dir, "ur10_robot.urdf")
            rmpflow_config_path = os.path.join(rmp_config_dir, "rmpflow", "ur10_rmpflow_config.yaml")
            self.arm_dof_num = 6
            self.ee_frame_name = "ee_link"
        else:
            raise NotImplementedError(f"Input robot has no configuration: {robot_name}")

        print(f"[INFO]: Loading controller URDF from: {robot_urdf_path}")
        # find all prims
        self._prim_paths = find_matching_prim_paths(prim_paths_expr)
        self.num_robots = len(self._prim_paths)
        # create all franka robots references and their controllers
        self.articulation_policies = list()
        for prim_path in self._prim_paths:
            # add robot reference
            robot = Articulation(prim_path)
            robot.initialize()
            # add controller
            rmpflow = RmpFlow(
                robot_description_path=robot_description_path,
                urdf_path=robot_urdf_path,
                rmpflow_config_path=rmpflow_config_path,
                end_effector_frame_name=self.ee_frame_name,
                evaluations_per_frame=5,
            )
            # wrap rmpflow to connect to the Franka robot articulation
            articulation_policy = ArticulationMotionPolicy(robot, rmpflow, dt)
            self.articulation_policies.append(articulation_policy)
        # create buffers
        # -- for storing command
        self.desired_ee_pos = None
        self.desired_ee_rot = None
        # -- for policy output
        self.joint_positions = torch.zeros((self.num_robots, self.arm_dof_num), device=self._device)
        self.joint_velocities = torch.zeros((self.num_robots, self.arm_dof_num), device=self._device)
        # update timer
        self._update_timer = torch.zeros(self.num_robots, device=self._device)

    """
    Operations.
    """

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        if robot_ids is None:
            robot_ids = ...
        self._update_timer[robot_ids] = 0.0

    def set_command(self, desired_ee_pos: torch.Tensor, desired_ee_rot: torch.Tensor):
        """Set target end-effector pose command."""
        # convert pose to numpy
        self.desired_ee_pos = desired_ee_pos.cpu().numpy()
        self.desired_ee_rot = desired_ee_rot.cpu().numpy()

    def advance(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target joint positions and velocity commands.
        """
        # check valid
        if self.desired_ee_pos is None or self.desired_ee_rot is None:
            raise RuntimeError("Target command has not been set. Please call `set_command()` first.")
        # update timer
        self._update_timer += dt
        # compute control actions
        for i, policy in enumerate(self.articulation_policies):
            # set rmpflow target to be the current position of the target cube.
            policy.get_motion_policy().set_end_effector_target(
                target_position=self.desired_ee_pos[i], target_orientation=self.desired_ee_rot[i]
            )
            # apply action on the robot
            action = policy.get_next_articulation_action()
            # copy actions into buffer
            # arm-action
            for dof_index in range(self.arm_dof_num):
                if action.joint_positions[dof_index] is not None:
                    self.joint_positions[i, dof_index] = torch.tensor(
                        action.joint_positions[dof_index], device=self._device
                    )
                if action.joint_velocities[dof_index] is not None:
                    self.joint_velocities[i, dof_index] = torch.tensor(
                        action.joint_velocities[dof_index], device=self._device
                    )
        return self.joint_positions, self.joint_velocities
