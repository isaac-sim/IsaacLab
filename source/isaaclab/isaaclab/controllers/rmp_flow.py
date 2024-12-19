# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims.articulations import Articulation
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed

from isaaclab.utils import configclass


@configclass
class RmpFlowControllerCfg:
    """Configuration for RMP-Flow controller (provided through LULA library)."""

    name: str = "rmp_flow"
    """Name of the controller. Supported: "rmp_flow", "rmp_flow_smoothed". Defaults to "rmp_flow"."""
    config_file: str = MISSING
    """Path to the configuration file for the controller."""
    urdf_file: str = MISSING
    """Path to the URDF model of the robot."""
    collision_file: str = MISSING
    """Path to collision model description of the robot."""
    frame_name: str = MISSING
    """Name of the robot frame for task space (must be present in the URDF)."""
    evaluations_per_frame: float = MISSING
    """Number of substeps during Euler integration inside LULA world model."""
    ignore_robot_state_updates: bool = False
    """If true, then state of the world model inside controller is rolled out. Defaults to False."""


class RmpFlowController:
    """Wraps around RMPFlow from IsaacSim for batched environments."""

    def __init__(self, cfg: RmpFlowControllerCfg, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            device: The device to use for computation.
        """
        # store input
        self.cfg = cfg
        self._device = device
        # display info
        print(f"[INFO]: Loading RMPFlow controller URDF from: {self.cfg.urdf_file}")

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

    def initialize(self, prim_paths_expr: str):
        """Initialize the controller.

        Args:
            prim_paths_expr: The expression to find the articulation prim paths.
        """
        # obtain the simulation time
        physics_dt = SimulationContext.instance().get_physics_dt()
        # find all prims
        self._prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
        self.num_robots = len(self._prim_paths)
        # resolve controller
        if self.cfg.name == "rmp_flow":
            controller_cls = RmpFlow
        elif self.cfg.name == "rmp_flow_smoothed":
            controller_cls = RmpFlowSmoothed
        else:
            raise ValueError(f"Unsupported controller in Lula library: {self.cfg.name}")
        # create all franka robots references and their controllers
        self.articulation_policies = list()
        for prim_path in self._prim_paths:
            # add robot reference
            robot = Articulation(prim_path)
            robot.initialize()
            # add controller
            rmpflow = controller_cls(
                robot_description_path=self.cfg.collision_file,
                urdf_path=self.cfg.urdf_file,
                rmpflow_config_path=self.cfg.config_file,
                end_effector_frame_name=self.cfg.frame_name,
                maximum_substep_size=physics_dt / self.cfg.evaluations_per_frame,
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

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        # if no robot ids are provided, then reset all robots
        if robot_ids is None:
            robot_ids = torch.arange(self.num_robots, device=self._device)
        # reset policies for specified robots
        for index in robot_ids:
            self.articulation_policies[index].motion_policy.reset()

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        # store command
        self._command[:] = command

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Returns:
            The target joint positions and velocity commands.
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
            self.dof_pos_target[i, :] = torch.from_numpy(action.joint_positions[:]).to(self.dof_pos_target)
            self.dof_vel_target[i, :] = torch.from_numpy(action.joint_velocities[:]).to(self.dof_vel_target)

        return self.dof_pos_target, self.dof_vel_target
