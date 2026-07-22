# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from .joint_impedance_cfg import JointImpedanceControllerCfg


class JointImpedanceController:
    """Joint impedance regulation control.

    The impedance (proportional-derivative plus optional inertial and gravity compensation) law is
    evaluated by Newton's model-free joint-impedance controller
    (:class:`newton.controllers.ControllerJointImpedanceModelFree`). Command shaping (absolute/relative
    targets, position offsets, and joint-limit clamping) and the gain schedule remain in Isaac Lab so
    the public configuration, command, and output contracts are preserved. Solves run through float32
    internal buffers.

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

        # build the Newton controller backend and the persistent Torch/Warp bridge buffers
        self._initialize_controller()

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
            stiffness = stiffness.clip_(min=self._p_gains_limits[..., 0], max=self._p_gains_limits[..., 1])
            # joint positions + stiffness
            self._dof_pos_target[:] = dof_pos_command
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains)  # critically damped
        elif self.cfg.impedance_mode == "variable":
            # split input command
            dof_pos_command, stiffness, damping_ratio = torch.tensor_split(command, 3, dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self._p_gains_limits[..., 0], max=self._p_gains_limits[..., 1])
            damping_ratio = damping_ratio.clip_(
                min=self._damping_ratio_limits[..., 0], max=self._damping_ratio_limits[..., 1]
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
        # resolve the command type into an absolute desired joint position
        if self.cfg.command_type == "p_abs":
            desired_dof_pos = self._dof_pos_target + self._dof_pos_offset
        elif self.cfg.command_type == "p_rel":
            desired_dof_pos = self._dof_pos_target + dof_pos
        else:
            raise ValueError(f"Invalid dof position command mode: {self.cfg.command_type}.")
        # clamp the desired position to the joint limits before handing it to the solver
        desired_dof_pos = desired_dof_pos.clip(min=self._dof_pos_limits[..., 0], max=self._dof_pos_limits[..., 1])

        # fill the persistent bridge buffers in place so the Warp views observe the new data
        self._joint_q_des.copy_(desired_dof_pos)
        self._joint_q.copy_(dof_pos)
        self._joint_qd.copy_(dof_vel)
        if self.cfg.inertial_compensation:
            self._mass_matrix.copy_(mass_matrix)
        if self.cfg.gravity_compensation:
            self._gravity.copy_(gravity)

        # evaluate the impedance law on the Newton backend and return the torque view
        self._controller.compute(self._controller_input, self._controller_output, None, None, self._time_step)
        return self._joint_f

    """
    Internal helpers.
    """

    def _initialize_controller(self) -> None:
        """Construct the Newton controller and wire the persistent Torch/Warp bridge buffers.

        The import is deferred to construction so importing this module never requires Newton; only
        instantiating the controller does.
        """
        from newton.controllers import ControllerJointImpedanceModelFree

        num_robots, num_dof = self.num_robots, self.num_dof
        total_dofs = num_robots * num_dof

        # homogeneous fleet: every robot exposes the same, contiguous block of DOFs
        dofs_per_robot = wp.array(np.full(num_robots, num_dof, dtype=np.int32), dtype=wp.int32, device=self._device)
        default_dof_indices = wp.array(np.arange(total_dofs, dtype=np.uint32), dtype=wp.uint32, device=self._device)

        # gains are live input ports (updated per-step for the variable impedance modes)
        self._controller = ControllerJointImpedanceModelFree(
            num_robots=num_robots,
            dofs_per_robot=dofs_per_robot,
            max_dofs=num_dof,
            default_dof_indices=default_dof_indices,
            stiffness="stiffness",
            damping="damping",
            use_gravity_compensation=self.cfg.gravity_compensation,
            use_coriolis_compensation=False,
            use_inertia_decoupling=self.cfg.inertial_compensation,
            has_qdd_feedforward=False,
            device=self._device,
        )
        self._controller_input = self._controller.input()
        self._controller_output = self._controller.output()

        # persistent bridge buffers: 2-D Torch tensors flattened into the controller's 1-D DOF ports
        self._joint_q = torch.zeros(num_robots, num_dof, device=self._device)
        self._joint_qd = torch.zeros(num_robots, num_dof, device=self._device)
        self._joint_q_des = torch.zeros(num_robots, num_dof, device=self._device)
        # desired velocity is always zero (Isaac Lab regulates about zero joint velocity)
        self._joint_qd_des = torch.zeros(num_robots, num_dof, device=self._device)
        self._controller_input.joint_q = wp.from_torch(self._joint_q.view(-1))
        self._controller_input.joint_qd = wp.from_torch(self._joint_qd.view(-1))
        self._controller_input.joint_q_des = wp.from_torch(self._joint_q_des.view(-1))
        self._controller_input.joint_qd_des = wp.from_torch(self._joint_qd_des.view(-1))

        # gains bind directly to the schedule buffers so ``set_command`` updates propagate in place
        self._controller_input.stiffness = wp.from_torch(self._p_gains)
        self._controller_input.damping = wp.from_torch(self._d_gains)

        if self.cfg.gravity_compensation:
            self._gravity = torch.zeros(num_robots, num_dof, device=self._device)
            self._controller_input.gravity_force = wp.from_torch(self._gravity.view(-1))
        if self.cfg.inertial_compensation:
            self._mass_matrix = torch.zeros(num_robots, num_dof, num_dof, device=self._device)
            self._controller_input.mass_matrix = wp.from_torch(self._mass_matrix)

        # torque output aliases the controller's flat output port, reshaped to (num_robots, num_dof)
        self._joint_f = wp.to_torch(self._controller_output.joint_f).view(num_robots, num_dof)
        self._time_step = wp.ones(1, dtype=wp.float32, device=self._device)
