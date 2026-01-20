# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from .lee_acceleration_control_cfg import LeeAccControllerCfg
from .lee_controller_base import LeeControllerBase
from .lee_controller_utils import compute_body_torque, compute_desired_orientation

if TYPE_CHECKING:
    from isaaclab.assets import Multirotor


class LeeAccController(LeeControllerBase):
    """Lee acceleration controller for multirotor tracking acceleration setpoints.

    Computes a body-frame wrench command ``[Fx, Fy, Fz, Tx, Ty, Tz]`` from an acceleration setpoint
    in the world frame. Gains may be randomized per environment if enabled in the configuration.
    """

    cfg: LeeAccControllerCfg

    def __init__(self, cfg: LeeAccControllerCfg, asset: Multirotor, num_envs: int, device: str):
        """Initialize controller.

        Args:
            cfg: Controller configuration.
            asset: Multirotor asset to control.
            num_envs: Number of environments.
            device: Device to run computations on.
        """
        super().__init__(cfg, asset, num_envs, device)

        # Gain ranges
        self.K_rot_range = torch.tensor(self.cfg.K_rot_range, device=device).repeat(num_envs, 1, 1)
        self.K_angvel_range = torch.tensor(self.cfg.K_angvel_range, device=device).repeat(num_envs, 1, 1)

        # Current gains
        self.K_rot_current = self.K_rot_range.mean(dim=1)
        self.K_angvel_current = self.K_angvel_range.mean(dim=1)

    def compute(self, command: torch.Tensor) -> torch.Tensor:
        """Compute wrench command from acceleration setpoint.

        Args:
            command: (num_envs, 3) acceleration command in world frame [m/s^2].

        Returns:
            (num_envs, 6) wrench command [fx, fy, fz, tx, ty, tz] in body frame.
        """
        self.wrench_command_b.zero_()

        # Use command directly as acceleration setpoint
        acc = command[:, :3].clone()
        forces_w = (acc - self.gravity) * self.mass.view(-1, 1)

        # Project forces to body z-axis for thrust command
        body_z_w = math_utils.matrix_from_quat(self.robot.data.root_quat_w)[:, :, 2]
        self.wrench_command_b[:, 2] = torch.sum(forces_w * body_z_w, dim=1)

        # Get current yaw and compute desired orientation
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        desired_quat = compute_desired_orientation(forces_w, yaw, self.rotation_matrix_buffer)

        # Zero angular velocity setpoint (hover)
        desired_angvel_b = torch.zeros((self.num_envs, 3), device=self.device)

        # Compute torque command
        self.wrench_command_b[:, 3:6] = compute_body_torque(
            desired_quat,
            desired_angvel_b,
            self.robot.data.root_quat_w,
            self.robot.data.root_ang_vel_b,
            self.robot_inertia,
            self.K_rot_current,
            self.K_angvel_current,
            self.cfg.max_yaw_rate,
        )

        return self.wrench_command_b

    def _randomize_params(self, env_ids: slice | torch.Tensor):
        """Randomize controller gains for the given environments if enabled."""
        if not self.cfg.randomize_params:
            return
        self.K_rot_current[env_ids] = math_utils.sample_uniform(
            self.K_rot_range[env_ids, 0], self.K_rot_range[env_ids, 1], self.K_rot_range[env_ids, 0].shape, self.device
        )
        self.K_angvel_current[env_ids] = math_utils.sample_uniform(
            self.K_angvel_range[env_ids, 0],
            self.K_angvel_range[env_ids, 1],
            self.K_angvel_range[env_ids, 0].shape,
            self.device,
        )
