# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.sim import SimulationContext


class G1LowerBodyStandingMotionControllerRetargeter(RetargeterBase):
    """Provides lower body standing commands for the G1 robot."""

    def __init__(self, cfg: G1LowerBodyStandingMotionControllerRetargeterCfg):
        """Initialize the retargeter."""
        super().__init__(cfg)
        self.cfg = cfg
        self._hip_height = cfg.hip_height

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

        # Thumbstick values are in the range of [-1, 1], so we need to scale them to the range of
        # [-movement_scale, movement_scale]
        left_thumbstick_x = left_thumbstick_x * self.cfg.movement_scale
        left_thumbstick_y = left_thumbstick_y * self.cfg.movement_scale

        # Use rendering time step for deterministic hip height adjustment regardless of wall clock time.
        dt = SimulationContext.instance().get_rendering_dt()
        self._hip_height -= right_thumbstick_y * dt * self.cfg.rotation_scale
        self._hip_height = max(0.4, min(1.0, self._hip_height))

        return torch.tensor(
            [-left_thumbstick_y, -left_thumbstick_x, -right_thumbstick_x, self._hip_height],
            device=self.cfg.sim_device,
            dtype=torch.float32,
        )

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class G1LowerBodyStandingMotionControllerRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body standing retargeter."""

    hip_height: float = 0.72
    """Height of the G1 robot hip in meters. The value is a fixed height suitable for G1 to do tabletop manipulation."""

    movement_scale: float = 0.5
    """Scale the movement of the robot to the range of [-movement_scale, movement_scale]."""

    rotation_scale: float = 0.35
    """Scale the rotation of the robot to the range of [-rotation_scale, rotation_scale]."""
    retargeter_type: type[RetargeterBase] = G1LowerBodyStandingMotionControllerRetargeter
