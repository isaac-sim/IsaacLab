# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.sim import SimulationContext


class LocomotionRootCmdRetargeter(RetargeterBase):
    """Provides root velocity and hip height commands for locomotion.

    This retargeter maps motion controller thumbsticks to:
    - Linear velocity (X, Y)
    - Angular velocity (Z)
    - Hip height adjustment
    """

    def __init__(self, cfg: LocomotionRootCmdRetargeterCfg):
        """Initialize the retargeter."""
        super().__init__(cfg)
        self.cfg = cfg
        self._hip_height = cfg.initial_hip_height

    def retarget(self, data: dict) -> torch.Tensor:
        left_thumbstick_x = 0.0
        left_thumbstick_y = 0.0
        right_thumbstick_x = 0.0
        right_thumbstick_y = 0.0

        # Get controller data using enums
        if (
            DeviceBase.TrackingTarget.CONTROLLER_LEFT in data
            and data[DeviceBase.TrackingTarget.CONTROLLER_LEFT] is not None
        ):
            left_controller_data = data[DeviceBase.TrackingTarget.CONTROLLER_LEFT]
            if len(left_controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
                left_inputs = left_controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
                if len(left_inputs) > DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value:
                    left_thumbstick_x = left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value]
                    left_thumbstick_y = left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value]

        if (
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT in data
            and data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] is not None
        ):
            right_controller_data = data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT]
            if len(right_controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
                right_inputs = right_controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
                if len(right_inputs) > DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value:
                    right_thumbstick_x = right_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value]
                    right_thumbstick_y = right_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value]

        # Thumbstick values are in the range of [-1, 1], so we need to scale them
        left_thumbstick_x = left_thumbstick_x * self.cfg.movement_scale
        left_thumbstick_y = left_thumbstick_y * self.cfg.movement_scale

        # Use rendering time step for deterministic hip height adjustment regardless of wall clock time.
        dt = SimulationContext.instance().get_rendering_dt()
        self._hip_height -= right_thumbstick_y * dt * self.cfg.rotation_scale
        self._hip_height = max(0.4, min(1.0, self._hip_height))

        # Returns [vel_x, vel_y, rot_vel_z, hip_height]
        # Note: left_thumbstick_y is forward/backward, so it maps to X velocity (negated because up is +1)
        #       left_thumbstick_x is left/right, so it maps to Y velocity (negated because right is +1)
        #       right_thumbstick_x is rotation, so it maps to Z angular velocity (negated)
        return torch.tensor(
            [-left_thumbstick_y, -left_thumbstick_x, -right_thumbstick_x, self._hip_height],
            device=self.cfg.sim_device,
            dtype=torch.float32,
        )

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class LocomotionRootCmdRetargeterCfg(RetargeterCfg):
    """Configuration for the locomotion root command retargeter."""

    initial_hip_height: float = 0.72
    """Initial height of the robot hip in meters."""

    movement_scale: float = 0.5
    """Scale the movement of the robot to the range of [-movement_scale, movement_scale]."""

    rotation_scale: float = 0.35
    """Scale the rotation of the robot to the range of [-rotation_scale, rotation_scale]."""

    retargeter_type: type[RetargeterBase] = LocomotionRootCmdRetargeter
