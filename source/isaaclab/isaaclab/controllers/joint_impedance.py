# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from .joint_impedance_cfg import JointImpedanceControllerCfg


class JointImpedanceController:
    """Joint impedance regulation control.

    Reference:
        [1] https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
    """

    def __init__(self, cfg: JointImpedanceControllerCfg, num_robots: int, dof_pos_limits: torch.Tensor, device: str):
        """Initialize joint impedance controller.

        Args:
            cfg: The configuration for the controller.
            num_robots: The number of robots to control.
            dof_pos_limits: The joint position limits for each robot. This is a tensor of shape
                (num_robots, num_dof, 2) where the last dimension contains the lower and upper limits.
            device: The device to use for computations.

        Raises:
            ValueError: When the shape of :obj:`dof_pos_limits` is not (num_robots, num_dof, 2).
        """
        self._controller = None
        # check valid inputs
        if len(dof_pos_limits.shape) != 3:
            raise ValueError(f"Joint position limits has shape '{dof_pos_limits.shape}'. Expected length of shape = 3.")
        # store inputs
        self.cfg = cfg
        self.num_robots = num_robots
        self.num_dof = dof_pos_limits.shape[1]  # (num_robots, num_dof, 2)
        self._device = device

        # create buffers
        # -- commands
        self._dof_pos_target = torch.zeros(self.num_robots, self.num_dof, device=self._device)
        # -- offsets
        self._dof_pos_offset = torch.zeros(self.num_robots, self.num_dof, device=self._device)
        # -- limits
        self._dof_pos_limits = dof_pos_limits
        # -- positional gains
        self._p_gains = torch.zeros(self.num_robots, self.num_dof, device=self._device)
        self._p_gains[:] = torch.tensor(self.cfg.stiffness, device=self._device)
        # -- velocity gains
        self._d_gains = torch.zeros(self.num_robots, self.num_dof, device=self._device)
        self._d_gains[:] = 2 * torch.sqrt(self._p_gains) * torch.tensor(self.cfg.damping_ratio, device=self._device)
        # -- position offsets
        if self.cfg.dof_pos_offset is not None:
            self._dof_pos_offset[:] = torch.tensor(self.cfg.dof_pos_offset, device=self._device)

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # joint positions
            return self.num_dof
        elif self.cfg.impedance_mode == "variable_kp":
            # joint positions + stiffness
            return self.num_dof * 2
        elif self.cfg.impedance_mode == "variable":
            # joint positions + stiffness + damping
            return self.num_dof * 3
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

    """
    Operations.
    """

    def initialize(self):
        """Initialize the internals."""
        pass

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        pass

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command.

        Args:
            command: The command to set. This is a tensor of shape (num_robots, num_actions) where
                :obj:`num_actions` is the dimension of the action space of the controller.
        """
        # check input size
        if command.shape != (self.num_robots, self.num_actions):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_robots, self.num_actions)}'."
            )
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # joint positions
            self._dof_pos_target[:] = command
        elif self.cfg.impedance_mode == "variable_kp":
            # split input command
            dof_pos_command, stiffness = torch.tensor_split(command, 2, dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self.cfg.stiffness_limits[0], max=self.cfg.stiffness_limits[1])
            # joint positions + stiffness
            self._dof_pos_target[:] = dof_pos_command
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains)  # critically damped
        elif self.cfg.impedance_mode == "variable":
            # split input command
            dof_pos_command, stiffness, damping_ratio = torch.tensor_split(command, 3, dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self.cfg.stiffness_limits[0], max=self.cfg.stiffness_limits[1])
            damping_ratio = damping_ratio.clip_(
                min=self.cfg.damping_ratio_limits[0], max=self.cfg.damping_ratio_limits[1]
            )
            # joint positions + stiffness + damping
            self._dof_pos_target[:] = dof_pos_command
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains) * damping_ratio
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

    def compute(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            dof_pos: The current joint positions.
            dof_vel: The current joint velocities.
            mass_matrix: The joint-space inertial matrix. Defaults to None.
            gravity: The joint-space gravity vector. Defaults to None.

        Raises:
            ValueError: When the command type is invalid.

        Returns:
            The target joint torques commands.
        """
        # resolve the command type
        if self.cfg.command_type == "p_abs":
            desired_dof_pos = self._dof_pos_target + self._dof_pos_offset
        elif self.cfg.command_type == "p_rel":
            desired_dof_pos = self._dof_pos_target + dof_pos
        else:
            raise ValueError(f"Invalid dof position command mode: {self.cfg.command_type}.")
        # compute errors
        desired_dof_pos = desired_dof_pos.clip_(min=self._dof_pos_limits[..., 0], max=self._dof_pos_limits[..., 1])
        if self.cfg.use_newton:
            return self._compute_newton(desired_dof_pos, dof_pos, dof_vel, mass_matrix, gravity)
        dof_pos_error = desired_dof_pos - dof_pos
        dof_vel_error = -dof_vel
        # compute acceleration
        des_dof_acc = self._p_gains * dof_pos_error + self._d_gains * dof_vel_error
        # compute torques
        # -- inertial compensation
        if self.cfg.inertial_compensation:
            # inverse dynamics control
            desired_torques = (mass_matrix @ des_dof_acc.unsqueeze(-1)).squeeze(-1)
        else:
            # decoupled spring-mass control
            desired_torques = des_dof_acc
        # -- gravity compensation (bias correction)
        if self.cfg.gravity_compensation:
            desired_torques += gravity

        return desired_torques

    def _compute_newton(
        self,
        desired_dof_pos: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        mass_matrix: torch.Tensor | None,
        gravity: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate Newton with the shared commands and gains."""
        if self._controller is None:
            self._initialize_newton()
        # -- Newton input ports
        self._joint_q_des.copy_(desired_dof_pos)
        self._joint_q.copy_(dof_pos)
        self._joint_qd.copy_(dof_vel)
        if self.cfg.inertial_compensation:
            self._mass_matrix.copy_(mass_matrix)
        if self.cfg.gravity_compensation:
            self._gravity.copy_(gravity)
        else:
            self._gravity.zero_()

        # -- solve and return an independent snapshot (dt is unused)
        self._controller.step(inputs=self._controller_input, outputs=self._controller_output, dt=0.0)
        return self._joint_f.to(
            dtype=torch.promote_types(torch.float32, torch.promote_types(dof_pos.dtype, dof_vel.dtype)), copy=True
        )

    def _initialize_newton(self) -> None:
        """Construct Newton ports and expose their arrays as Torch views."""
        from newton.controllers import ControllerJointImpedanceModelFree

        # -- construct Newton controller and ports
        # Gains remain live inputs for variable impedance modes.
        self._controller = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=wp.full(self.num_robots, self.num_dof, dtype=wp.int32, device=self._device),
            stiffness=None,
            damping=None,
            use_gravity_compensation=True,
            use_coriolis_compensation=False,
            use_inertia_decoupling=self.cfg.inertial_compensation,
            has_qdd_feedforward=False,
            device=self._device,
        )
        self._controller_input = self._controller.input()
        self._controller_output = self._controller.output()

        self._controller_input.stiffness = wp.from_torch(self._p_gains.flatten())
        self._controller_input.damping = wp.from_torch(self._d_gains.flatten())
        self._joint_q = wp.to_torch(self._controller_input.joint_q).view(self.num_robots, self.num_dof)
        self._joint_qd = wp.to_torch(self._controller_input.joint_qd).view(self.num_robots, self.num_dof)
        self._joint_q_des = wp.to_torch(self._controller_input.joint_q_des).view(self.num_robots, self.num_dof)
        self._gravity = wp.to_torch(self._controller_input.gravity_force).view(self.num_robots, self.num_dof)
        if self.cfg.inertial_compensation:
            self._mass_matrix = wp.to_torch(self._controller_input.mass_matrix)
        self._joint_f = wp.to_torch(self._controller_output.joint_f).view(self.num_robots, self.num_dof)
