# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import numpy as np
import torch

# enable motion generation extensions
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

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
        self.cfg = cfg
        self._device = device
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

    def initialize(self, num_robots: int, joint_names: list[str]):
        """Initialize the controller.

        Args:
            num_robots: Number of robot instances (environments).
            joint_names: Ordered list of all joint names from the articulation.
        """
        physics_dt = sim_utils.SimulationContext.instance().get_physics_dt()
        self.num_robots = num_robots
        self._physics_dt = physics_dt

        if self.cfg.name == "rmp_flow":
            controller_cls = RmpFlow
        elif self.cfg.name == "rmp_flow_smoothed":
            controller_cls = RmpFlowSmoothed
        else:
            raise ValueError(f"Unsupported controller in Lula library: {self.cfg.name}")

        name_to_idx = {name: i for i, name in enumerate(joint_names)}

        self._rmpflow_policies: list[RmpFlow | RmpFlowSmoothed] = []
        self._active_indices: list[np.ndarray] = []
        self._watched_indices: list[np.ndarray] = []

        for _ in range(num_robots):
            local_urdf_file = retrieve_file_path(_resolve_rmpflow_path(self.cfg.urdf_file), force_download=True)
            local_collision_file = retrieve_file_path(
                _resolve_rmpflow_path(self.cfg.collision_file), force_download=True
            )
            local_config_file = retrieve_file_path(_resolve_rmpflow_path(self.cfg.config_file), force_download=True)

            rmpflow = controller_cls(
                robot_description_path=local_collision_file,
                urdf_path=local_urdf_file,
                rmpflow_config_path=local_config_file,
                end_effector_frame_name=self.cfg.frame_name,
                maximum_substep_size=physics_dt / self.cfg.evaluations_per_frame,
                ignore_robot_state_updates=self.cfg.ignore_robot_state_updates,
            )

            active_indices = np.array([name_to_idx[n] for n in rmpflow.get_active_joints()], dtype=np.intp)
            watched_indices = np.array([name_to_idx[n] for n in rmpflow.get_watched_joints()], dtype=np.intp)

            self._rmpflow_policies.append(rmpflow)
            self._active_indices.append(active_indices)
            self._watched_indices.append(watched_indices)

        self.active_dof_names = self._rmpflow_policies[0].get_active_joints()
        self.num_dof = len(self.active_dof_names)

        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device)
        self.dof_pos_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)
        self.dof_vel_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)

    def reset_idx(self, robot_ids: torch.Tensor | None = None):
        """Reset the internals."""
        if robot_ids is None:
            robot_ids = torch.arange(self.num_robots, device=self._device)
        for index in robot_ids:
            self._rmpflow_policies[index].reset()

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        self._command[:] = command

    def compute(
        self, joint_positions: torch.Tensor, joint_velocities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Args:
            joint_positions: Current joint positions, shape ``[num_robots, num_joints]``.
            joint_velocities: Current joint velocities, shape ``[num_robots, num_joints]``.

        Returns:
            The target joint positions and velocity commands.
        """
        command = self._command.cpu().numpy()
        all_pos = joint_positions.cpu().numpy()
        all_vel = joint_velocities.cpu().numpy()

        for i, rmpflow in enumerate(self._rmpflow_policies):
            rmpflow.set_end_effector_target(target_position=command[i, 0:3], target_orientation=command[i, 3:7])
            active_pos = all_pos[i][self._active_indices[i]]
            active_vel = all_vel[i][self._active_indices[i]]
            watched_pos = all_pos[i][self._watched_indices[i]]
            watched_vel = all_vel[i][self._watched_indices[i]]

            pos_targets, vel_targets = rmpflow.compute_joint_targets(
                active_pos, active_vel, watched_pos, watched_vel, self._physics_dt
            )
            self.dof_pos_target[i, :] = torch.from_numpy(pos_targets[:]).to(self.dof_pos_target)
            self.dof_vel_target[i, :] = torch.from_numpy(vel_targets[:]).to(self.dof_vel_target)

        return self.dof_pos_target, self.dof_vel_target
