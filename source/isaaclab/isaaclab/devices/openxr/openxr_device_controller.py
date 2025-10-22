# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction with motion controllers."""

import contextlib
import numpy as np
from dataclasses import dataclass
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase

from .openxr_device import OpenXRDevice, OpenXRDeviceCfg

with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRInputDevice

# Extend TrackingTarget enum for controllers
from enum import Enum

class MotionControllerInputIndex(Enum):
    """Enum for Motion Controller input indices."""
    THUMBSTICK_X = 0
    THUMBSTICK_Y = 1
    TRIGGER = 2
    SQUEEZE = 3
    BUTTON_0 = 4  # X for left controller, A for right controller
    BUTTON_1 = 5  # Y for left controller, B for right controller
    PADDING = 6 # Additional padding to make 7 elements to align with MotionControllerDataRowIndex.POSE


class MotionControllerDataRowIndex(Enum):
    """Enum for Motion Controller data row indices."""
    POSE = 0      # [x, y, z, w, x, y, z] - position and quaternion
    INPUTS = 1    # MotionControllerInputIndex: [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1]

# Create a new enum that includes all TrackingTarget values plus new ones
class MotionControllerTrackingTarget(Enum):
    """Extended tracking targets for Motion Controllers."""
    LEFT = len(OpenXRDevice.TrackingTarget)
    RIGHT = LEFT + 1


@dataclass
class OpenXRDeviceMotionControllerCfg(OpenXRDeviceCfg):
    """Configuration for Motion Controller OpenXR devices."""
    pass


class OpenXRDeviceMotionController(OpenXRDevice):

    def __init__(
        self,
        cfg: OpenXRDeviceMotionControllerCfg,
        retargeters: list[RetargeterBase] | None = None,
    ):
        """Initialize the OpenXR device.

        Args:
            cfg: Configuration object for OpenXR settings.
            retargeters: List of retargeter instances to use for transforming raw tracking data.
        """
        super().__init__(cfg, retargeters)

    """
    Operations
    """

    def reset(self):
        super().reset()

    def _get_raw_data(self) -> Any:
        """Get the latest tracking data from the OpenXR runtime.

        Returns:
            Dictionary with TrackingTarget enum keys containing:
                - HEAD: Single 7-element array with position and orientation
                - CONTROLLER_LEFT: 2D array [pose(7), inputs(7)]
                - CONTROLLER_RIGHT: 2D array [pose(7), inputs(7)]

        Controller data format:
            - Row 0 (pose): [x, y, z, w, x, y, z] - position and quaternion
            - Row 1 (inputs): [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, padding]

        Hand tracking data format:
            Each pose is represented as a 7-element array: [x, y, z, qw, qx, qy, qz]
            where the first 3 elements are position and the last 4 are quaternion orientation.
        """
        return {
            MotionControllerTrackingTarget.LEFT: self._query_controller(
                MotionControllerTrackingTarget.LEFT,
                XRCore.get_singleton().get_input_device("/user/hand/left")
            ),
            MotionControllerTrackingTarget.RIGHT: self._query_controller(
                MotionControllerTrackingTarget.RIGHT,
                XRCore.get_singleton().get_input_device("/user/hand/right")
            ),
            OpenXRDevice.TrackingTarget.HEAD: self._calculate_headpose(),
        }

    """
    Internal helpers.
    """

    def _query_controller(
        self, tracking_target : MotionControllerTrackingTarget, input_device
    ) -> np.array:
        """Calculate and update input device data

        """

        if input_device is None:
            return np.array([])

        pose = input_device.get_virtual_world_pose()
        position = pose.ExtractTranslation()
        quat = pose.ExtractRotationQuat()
        
        thumbstick_x = 0.0
        thumbstick_y = 0.0
        trigger = 0.0
        squeeze = 0.0
        button_0 = 0.0
        button_1 = 0.0

        if input_device.has_input_gesture("thumbstick", "x"):
            thumbstick_x: float = input_device.get_input_gesture_value("thumbstick", "x")

        if input_device.has_input_gesture("thumbstick", "y"):
            thumbstick_y: float = input_device.get_input_gesture_value("thumbstick", "y")

        if input_device.has_input_gesture("trigger", "value"):
            trigger: float = input_device.get_input_gesture_value("trigger", "value")

        if input_device.has_input_gesture("squeeze", "value"):
            squeeze: float = input_device.get_input_gesture_value("squeeze", "value")

        if tracking_target == MotionControllerTrackingTarget.LEFT:
            if input_device.has_input_gesture("x", "click"):
                button_0 = input_device.get_input_gesture_value("x", "click")

            if input_device.has_input_gesture("y", "click"):
                button_1 = input_device.get_input_gesture_value("y", "click")
        else:
            if input_device.has_input_gesture("a", "click"):
                button_0 = input_device.get_input_gesture_value("a", "click")

            if input_device.has_input_gesture("b", "click"):
                button_1 = input_device.get_input_gesture_value("b", "click")

        # First row: position and quaternion (7 values)
        pose_row = [
            position[0], position[1], position[2],  # x, y, z position
            quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]  # w, x, y, z quaternion
        ]
        
        # Second row: controller input values (6 values + 1 padding)
        input_row = [
            thumbstick_x,  # MotionControllerInputIndex.THUMBSTICK_X
            thumbstick_y,  # MotionControllerInputIndex.THUMBSTICK_Y
            trigger,       # MotionControllerInputIndex.TRIGGER
            squeeze,       # MotionControllerInputIndex.SQUEEZE
            button_0,      # MotionControllerInputIndex.BUTTON_0
            button_1,      # MotionControllerInputIndex.BUTTON_1
            0.0,           # MotionControllerInputIndex.PADDING
        ]
        
        # Combine into 2D array: [pose(7), inputs(7)]
        return np.array([pose_row, input_row], dtype=np.float32)
