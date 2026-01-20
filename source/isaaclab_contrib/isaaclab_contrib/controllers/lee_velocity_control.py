# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from .lee_controller_base import LeeControllerBase
from .lee_controller_utils import compute_body_torque, compute_desired_orientation, yaw_rate_to_body_angvel
from .lee_velocity_control_cfg import LeeVelControllerCfg

if TYPE_CHECKING:
    from isaaclab.assets import Multirotor


class LeeVelController(LeeControllerBase):
    """Lee velocity controller for multirotor tracking velocity setpoints.

    Computes a body-frame wrench command ``[Fx, Fy, Fz, Tx, Ty, Tz]`` from a velocity setpoint:
    [vx, vy, vz, yaw_rate]. Gains may be randomized per environment if enabled in the configuration.
    """

    cfg: LeeVelControllerCfg

    def __init__(self, cfg: LeeVelControllerCfg, asset: Multirotor, num_envs: int, device: str):
        """Initialize controller.

        Args:
            cfg: Controller configuration.
            asset: Multirotor asset to control.
            num_envs: Number of environments.
            device: Device to run computations on.
        """
        super().__init__(cfg, asset, num_envs, device)

        # Gain ranges
        self.K_vel_range = torch.tensor(self.cfg.K_vel_range, device=device).repeat(num_envs, 1, 1)
        self.K_rot_range = torch.tensor(self.cfg.K_rot_range, device=device).repeat(num_envs, 1, 1)
        self.K_angvel_range = torch.tensor(self.cfg.K_angvel_range, device=device).repeat(num_envs, 1, 1)

        # Current gains
        self.K_vel_current = self.K_vel_range.mean(dim=1)
        self.K_rot_current = self.K_rot_range.mean(dim=1)
        self.K_angvel_current = self.K_angvel_range.mean(dim=1)

    def compute(self, command: torch.Tensor) -> torch.Tensor:
        """Compute wrench command from velocity setpoint.

        Args:
            command: (num_envs, 4) velocity command [vx, vy, vz, yaw_rate] in body frame.

        Returns:
            (num_envs, 6) wrench command [fx, fy, fz, tx, ty, tz] in body frame.
        """
        self.wrench_command_b.zero_()

        # Compute acceleration from velocity tracking
        acc = self._compute_acceleration(setpoint_velocity=command[:, :3])
        forces_w = (acc - self.gravity) * self.mass.view(-1, 1)

        # Project forces to body z-axis for thrust command
        body_z_w = math_utils.matrix_from_quat(self.robot.data.root_quat_w)[:, :, 2]
        self.wrench_command_b[:, 2] = torch.sum(forces_w * body_z_w, dim=1)

        # Compute desired orientation from force direction and yaw setpoint
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        desired_quat = compute_desired_orientation(forces_w, yaw, self.rotation_matrix_buffer)

        # Compute desired angular velocity in body frame from yaw rate command
        desired_angvel_b = yaw_rate_to_body_angvel(command[:, 3], roll, pitch, self.device)

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
        self.K_vel_current[env_ids] = math_utils.sample_uniform(
            self.K_vel_range[env_ids, 0], self.K_vel_range[env_ids, 1], self.K_vel_range[env_ids, 0].shape, self.device
        )
        self.K_rot_current[env_ids] = math_utils.sample_uniform(
            self.K_rot_range[env_ids, 0], self.K_rot_range[env_ids, 1], self.K_rot_range[env_ids, 0].shape, self.device
        )
        self.K_angvel_current[env_ids] = math_utils.sample_uniform(
            self.K_angvel_range[env_ids, 0],
            self.K_angvel_range[env_ids, 1],
            self.K_angvel_range[env_ids, 0].shape,
            self.device,
        )

    def _compute_acceleration(self, setpoint_velocity: torch.Tensor) -> torch.Tensor:
        """Compute desired acceleration from velocity tracking error.

        Args:
            setpoint_velocity: (num_envs, 3) desired velocity in body frame.

        Returns:
            (num_envs, 3) desired acceleration in world frame.
        """
        # Get yaw-only orientation (vehicle frame)
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        vehicle_quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        # Transform setpoint from body to world frame
        setpoint_velocity_w = math_utils.quat_apply(vehicle_quat, setpoint_velocity)

        # Compute velocity error and acceleration command
        velocity_error = setpoint_velocity_w - self.robot.data.root_lin_vel_w
        return self.K_vel_current * velocity_error
