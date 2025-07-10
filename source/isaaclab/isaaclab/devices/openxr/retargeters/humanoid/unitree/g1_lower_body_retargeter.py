# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class G1LowerBodyRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body retargeter."""

    linear_velocity_scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for converting delta position to linear velocity command [x, y]."""
    angular_velocity_scale: float = 1.0
    """Scale factor for converting delta yaw to angular velocity command."""
    linear_velocity_threshold: float = 0.01
    """Threshold for linear displacement below which the velocity is zero (in meters)."""
    angular_velocity_threshold: float = 0.02
    """Threshold for angular displacement below which the velocity is zero (in radians)."""


class G1LowerBodyRetargeter(RetargeterBase):
    """Retargets OpenXR head tracking data to lower body velocity commands for the G1 robot."""

    def __init__(self, cfg: G1LowerBodyRetargeterCfg):
        """Initialize the retargeter."""
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._neutral_pose: torch.Tensor | None = None
        self._command_tensor = torch.zeros(4, device=self._sim_device)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert head pose to lower body velocity commands."""
        current_pose_np = data[OpenXRDevice.TrackingTarget.HEAD]
        current_pose = torch.tensor(current_pose_np, dtype=torch.float32, device=self._sim_device)

        if self._neutral_pose is None:
            self._neutral_pose = current_pose.clone()
            return self._command_tensor

        # Calculate displacement from neutral pose
        delta_pos_xy = current_pose[:2] - self._neutral_pose[:2]

        # Calculate delta yaw
        current_quat = current_pose[3:]
        neutral_quat = self._neutral_pose[3:]
        # The quaternion from OpenXR is expected to be in (w, x, y, z) format
        current_yaw = torch.atan2(
            2.0 * (current_quat[0] * current_quat[3] + current_quat[1] * current_quat[2]),
            1.0 - 2.0 * (current_quat[2] ** 2 + current_quat[3] ** 2),
        )
        neutral_yaw = torch.atan2(
            2.0 * (neutral_quat[0] * neutral_quat[3] + neutral_quat[1] * neutral_quat[2]),
            1.0 - 2.0 * (neutral_quat[2] ** 2 + neutral_quat[3] ** 2),
        )
        delta_yaw = current_yaw - neutral_yaw
        delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        # Apply thresholds (deadzone) and scaling
        linear_displacement = torch.norm(delta_pos_xy)
        if linear_displacement > self.cfg.linear_velocity_threshold:
            linear_velocity_xy = delta_pos_xy * torch.tensor(
                self.cfg.linear_velocity_scale, device=self._sim_device, dtype=torch.float32
            )
        else:
            linear_velocity_xy = torch.zeros(2, device=self._sim_device)

        if torch.abs(delta_yaw) > self.cfg.angular_velocity_threshold:
            angular_velocity_z = delta_yaw * self.cfg.angular_velocity_scale
        else:
            angular_velocity_z = torch.tensor(0.0, device=self._sim_device)

        # Extract absolute height
        h = current_pose[2]

        # Create command vector and update command tensor
        command_vector = torch.cat([linear_velocity_xy, angular_velocity_z.unsqueeze(0), h.unsqueeze(0)])
        self._command_tensor = command_vector

        return self._command_tensor
