# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
import torch
from dataclasses import dataclass

from isaaclab.devices.openxr.openxr_device_controller import (
    MotionControllerDataRowIndex,
    MotionControllerInputIndex,
    MotionControllerTrackingTarget,
)
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class G1LowerBodyStandingRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body standing retargeter."""

    hip_height: float = 0.72
    """Height of the G1 robot hip in meters. The value is a fixed height suitable for G1 to do tabletop manipulation."""


class G1LowerBodyStandingRetargeter(RetargeterBase):
    """Provides lower body standing commands for the G1 robot."""

    def __init__(self, cfg: G1LowerBodyStandingRetargeterCfg):
        """Initialize the retargeter."""
        self.cfg = cfg

    def retarget(self, data: dict) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device)


@dataclass
class G1LowerBodyStandingMotionControllerRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body standing retargeter."""

    hip_height: float = 0.72
    """Height of the G1 robot hip in meters. The value is a fixed height suitable for G1 to do tabletop manipulation."""

    movement_scale: float = 0.45
    """Scale the movement of the robot to the range of [-movement_scale, movement_scale]."""

    rotation_scale: float = 0.35
    """Scale the rotation of the robot to the range of [-rotation_scale, rotation_scale]."""


class G1LowerBodyStandingMotionControllerRetargeter(RetargeterBase):
    """Provides lower body standing commands for the G1 robot."""

    def __init__(self, cfg: G1LowerBodyStandingMotionControllerRetargeterCfg):
        """Initialize the retargeter."""
        self.cfg = cfg
        self._hip_height = cfg.hip_height
        self._last_update_time = time.time()

    def retarget(self, data: dict) -> torch.Tensor:
        left_thumbstick_x = 0.0
        left_thumbstick_y = 0.0
        right_thumbstick_x = 0.0
        right_thumbstick_y = 0.0

        # Get controller data using enums
        if MotionControllerTrackingTarget.LEFT in data and data[MotionControllerTrackingTarget.LEFT] is not None:
            left_controller_data = data[MotionControllerTrackingTarget.LEFT]
            if len(left_controller_data) > MotionControllerDataRowIndex.INPUTS.value:
                left_inputs = left_controller_data[MotionControllerDataRowIndex.INPUTS.value]
                if len(left_inputs) > MotionControllerInputIndex.THUMBSTICK_Y.value:
                    left_thumbstick_x = left_inputs[MotionControllerInputIndex.THUMBSTICK_X.value]
                    left_thumbstick_y = left_inputs[MotionControllerInputIndex.THUMBSTICK_Y.value]

        if MotionControllerTrackingTarget.RIGHT in data and data[MotionControllerTrackingTarget.RIGHT] is not None:
            right_controller_data = data[MotionControllerTrackingTarget.RIGHT]
            if len(right_controller_data) > MotionControllerDataRowIndex.INPUTS.value:
                right_inputs = right_controller_data[MotionControllerDataRowIndex.INPUTS.value]
                if len(right_inputs) > MotionControllerInputIndex.THUMBSTICK_Y.value:
                    right_thumbstick_x = right_inputs[MotionControllerInputIndex.THUMBSTICK_X.value]
                    right_thumbstick_y = right_inputs[MotionControllerInputIndex.THUMBSTICK_Y.value]

        # Thumbstick values are in the range of [-1, 1], so we need to scale them to the range of [-movement_clamp, movement_clamp]
        left_thumbstick_x = left_thumbstick_x * self.cfg.movement_scale
        left_thumbstick_y = left_thumbstick_y * self.cfg.movement_scale

        # Use wall clock time for consistent hip height adjustment regardless of simulation speed
        current_time = time.time()
        dt = current_time - self._last_update_time
        self._last_update_time = current_time
        self._hip_height -= right_thumbstick_y * dt * self.cfg.rotation_scale

        return torch.tensor(
            [-left_thumbstick_y, -left_thumbstick_x, -right_thumbstick_x, self._hip_height],
            device=self.cfg.sim_device,
            dtype=torch.float32,
        )
