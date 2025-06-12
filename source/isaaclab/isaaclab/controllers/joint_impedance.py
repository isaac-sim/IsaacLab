# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class JointImpedanceControllerCfg:
    """Configuration for joint impedance regulation controller."""

    command_type: str = "p_abs"
    """Type of command: p_abs (absolute) or p_rel (relative)."""

    dof_pos_offset: Sequence[float] | None = None
    """Offset to DOF position command given to controller. (default: None).

    If None then position offsets are set to zero.
    """

    impedance_mode: str = MISSING
    """Type of gains: "fixed", "variable", "variable_kp"."""

    inertial_compensation: bool = False
    """Whether to perform inertial compensation (inverse dynamics)."""

    gravity_compensation: bool = False
    """Whether to perform gravity compensation."""

    stiffness: float | Sequence[float] = MISSING
    """The positional gain for determining desired torques based on joint position error."""

    damping_ratio: float | Sequence[float] | None = None
    """The damping ratio is used in-conjunction with positional gain to compute desired torques
    based on joint velocity error.

    The following math operation is performed for computing velocity gains:
        :math:`d_gains = 2 * sqrt(p_gains) * damping_ratio`.
    """

    stiffness_limits: tuple[float, float] = (0, 300)
    """Minimum and maximum values for positional gains.

    Note: Used only when :obj:`impedance_mode` is "variable" or "variable_kp".
    """

    damping_ratio_limits: tuple[float, float] = (0, 100)
    """Minimum and maximum values for damping ratios used to compute velocity gains.

    Note: Used only when :obj:`impedance_mode` is "variable".
    """


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
        # -- position gain limits
        self._p_gains_limits = torch.zeros_like(self._dof_pos_limits)
        self._p_gains_limits[..., 0] = self.cfg.stiffness_limits[0]
        self._p_gains_limits[..., 1] = self.cfg.stiffness_limits[1]
        # -- damping ratio limits
        self._damping_ratio_limits = torch.zeros_like(self._dof_pos_limits)
        self._damping_ratio_limits[..., 0] = self.cfg.damping_ratio_limits[0]
        self._damping_ratio_limits[..., 1] = self.cfg.damping_ratio_limits[1]

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
            stiffness = stiffness.clip_(min=self._p_gains_limits[0], max=self._p_gains_limits[1])
            # joint positions + stiffness
            self._dof_pos_target[:] = dof_pos_command
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains)  # critically damped
        elif self.cfg.impedance_mode == "variable":
            # split input command
            dof_pos_command, stiffness, damping_ratio = torch.tensor_split(command, 3, dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self._p_gains_limits[0], max=self._p_gains_limits[1])
            damping_ratio = damping_ratio.clip_(min=self._damping_ratio_limits[0], max=self._damping_ratio_limits[1])
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
        dof_pos_error = desired_dof_pos - dof_pos
        dof_vel_error = -dof_vel
        # compute acceleration
        des_dof_acc = self._p_gains * dof_pos_error + self._d_gains * dof_vel_error
        # compute torques
        # -- inertial compensation
        if self.cfg.inertial_compensation:
            # inverse dynamics control
            desired_torques = mass_matrix @ des_dof_acc
        else:
            # decoupled spring-mass control
            desired_torques = des_dof_acc
        # -- gravity compensation (bias correction)
        if self.cfg.gravity_compensation:
            desired_torques += gravity

        return desired_torques
