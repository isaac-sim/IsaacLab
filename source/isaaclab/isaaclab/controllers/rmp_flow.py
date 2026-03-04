# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch

from isaacsim.core.prims import SingleArticulation

# enable motion generation extensions
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path

from .rmp_flow_cfg import RmpFlowControllerCfg  # noqa: F401

_RMPFLOW_EXT_PREFIX = "rmpflow_ext:"
_RMPFLOW_EXT_NAME = "isaacsim.robot_motion.motion_generation"


def _resolve_rmpflow_path(path: str) -> str:
    """Resolve a sentinel ``rmpflow_ext:`` path to an absolute filesystem path.

    Paths stored in :class:`~isaaclab.controllers.rmp_flow_cfg.RmpFlowControllerCfg`
    that begin with ``"rmpflow_ext:"`` are relative to the
    ``isaacsim.robot_motion.motion_generation`` extension directory.  This avoids
    importing ``isaacsim`` in the cfg file (which is loaded without Kit).
    """
    if path.startswith(_RMPFLOW_EXT_PREFIX):
        rel = path[len(_RMPFLOW_EXT_PREFIX) :]
        ext_dir = get_extension_path_from_name(_RMPFLOW_EXT_NAME)
        return os.path.join(ext_dir, rel)
    return path


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
        physics_dt = sim_utils.SimulationContext.instance().get_physics_dt()
        # find all prims
        self._prim_paths = sim_utils.find_matching_prim_paths(prim_paths_expr)
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
            robot = SingleArticulation(prim_path)
            robot.initialize()
            # download files if they are not local

            local_urdf_file = retrieve_file_path(_resolve_rmpflow_path(self.cfg.urdf_file), force_download=True)
            local_collision_file = retrieve_file_path(
                _resolve_rmpflow_path(self.cfg.collision_file), force_download=True
            )
            local_config_file = retrieve_file_path(_resolve_rmpflow_path(self.cfg.config_file), force_download=True)

            # add controller
            rmpflow = controller_cls(
                robot_description_path=local_collision_file,
                urdf_path=local_urdf_file,
                rmpflow_config_path=local_config_file,
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
