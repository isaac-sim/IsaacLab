# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction."""

import contextlib
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import carb

from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase

from ..device_base import DeviceBase, DeviceCfg
from .xr_cfg import XrCfg

# For testing purposes, we need to mock the XRCore, XRPoseValidityFlags classes
XRCore = None
XRPoseValidityFlags = None

with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRPoseValidityFlags
from isaacsim.core.prims import SingleXFormPrim


@dataclass
class OpenXRDeviceCfg(DeviceCfg):
    """Configuration for OpenXR devices."""

    xr_cfg: XrCfg | None = None


class OpenXRDevice(DeviceBase):
    """An OpenXR-powered device for teleoperation and interaction.

    This device tracks hand joints using OpenXR and makes them available as:
    1. A dictionary of tracking data (when used without retargeters)
    2. Retargeted commands for robot control (when retargeters are provided)

    Raw data format (_get_raw_data output):
    * A dictionary with keys matching TrackingTarget enum values (HAND_LEFT, HAND_RIGHT, HEAD)
    * Each hand tracking entry contains a dictionary of joint poses
    * Each joint pose is a 7D vector (x, y, z, qw, qx, qy, qz) in meters and quaternion units
    * Joint names are defined in HAND_JOINT_NAMES from isaaclab.devices.openxr.common
    * Supported joints include palm, wrist, and joints for thumb, index, middle, ring and little fingers

    Teleop commands:
    The device responds to several teleop commands that can be subscribed to via add_callback():
    * "START": Resume hand tracking data flow
    * "STOP": Pause hand tracking data flow
    * "RESET": Reset the tracking and signal simulation reset

    The device tracks the left hand, right hand, head position, or any combination of these
    based on the TrackingTarget enum values. When retargeters are provided, the raw tracking
    data is transformed into robot control commands suitable for teleoperation.
    """

    class TrackingTarget(Enum):
        """Enum class specifying what to track with OpenXR.

        Attributes:
            HAND_LEFT: Track the left hand (index 0 in _get_raw_data output)
            HAND_RIGHT: Track the right hand (index 1 in _get_raw_data output)
            HEAD: Track the head/headset position (index 2 in _get_raw_data output)
        """

        HAND_LEFT = 0
        HAND_RIGHT = 1
        HEAD = 2

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"

    def __init__(
        self,
        cfg: OpenXRDeviceCfg,
        retargeters: list[RetargeterBase] | None = None,
    ):
        """Initialize the OpenXR device.

        Args:
            cfg: Configuration object for OpenXR settings.
            retargeters: List of retargeter instances to use for transforming raw tracking data.
        """
        super().__init__(retargeters)
        self._xr_cfg = cfg.xr_cfg or XrCfg()
        self._additional_callbacks = dict()
        self._vc_subscription = (
            XRCore.get_singleton()
            .get_message_bus()
            .create_subscription_to_pop_by_type(
                carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
            )
        )

        # Initialize dictionaries instead of arrays
        default_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_headpose = default_pose.copy()

        xr_anchor = SingleXFormPrim("/XRAnchor", position=self._xr_cfg.anchor_pos, orientation=self._xr_cfg.anchor_rot)
        carb.settings.get_settings().set_float("/persistent/xr/profile/ar/render/nearPlane", self._xr_cfg.near_plane)
        carb.settings.get_settings().set_string("/persistent/xr/profile/ar/anchorMode", "custom anchor")
        carb.settings.get_settings().set_string("/xrstage/profile/ar/customAnchor", xr_anchor.prim_path)

    def __del__(self):
        """Clean up resources when the object is destroyed.

        Properly unsubscribes from the XR message bus to prevent memory leaks
        and resource issues when the device is no longer needed.
        """
        if hasattr(self, "_vc_subscription") and self._vc_subscription is not None:
            self._vc_subscription = None

        # No need to explicitly clean up OpenXR instance as it's managed by NVIDIA Isaac Sim

    def __str__(self) -> str:
        """Returns a string containing information about the OpenXR hand tracking device.

        This provides details about the device configuration, tracking settings,
        and available gesture commands.

        Returns:
            Formatted string with device information
        """

        msg = f"OpenXR Hand Tracking Device: {self.__class__.__name__}\n"
        msg += f"\tAnchor Position: {self._xr_cfg.anchor_pos}\n"
        msg += f"\tAnchor Rotation: {self._xr_cfg.anchor_rot}\n"

        # Add retargeter information
        if self._retargeters:
            msg += "\tRetargeters:\n"
            for i, retargeter in enumerate(self._retargeters):
                msg += f"\t\t{i + 1}. {retargeter.__class__.__name__}\n"
        else:
            msg += "\tRetargeters: None (raw joint data output)\n"

        # Add available gesture commands
        msg += "\t----------------------------------------------\n"
        msg += "\tAvailable Gesture Commands:\n"

        # Check which callbacks are registered
        start_avail = "START" in self._additional_callbacks
        stop_avail = "STOP" in self._additional_callbacks
        reset_avail = "RESET" in self._additional_callbacks

        msg += f"\t\tStart Teleoperation: {'✓' if start_avail else '✗'}\n"
        msg += f"\t\tStop Teleoperation: {'✓' if stop_avail else '✗'}\n"
        msg += f"\t\tReset Environment: {'✓' if reset_avail else '✗'}\n"

        # Add joint tracking information
        msg += "\t----------------------------------------------\n"
        msg += "\tTracked Joints: All 26 XR hand joints including:\n"
        msg += "\t\t- Wrist, palm\n"
        msg += "\t\t- Thumb (tip, intermediate joints)\n"
        msg += "\t\t- Fingers (tip, distal, intermediate, proximal)\n"

        return msg

    """
    Operations
    """

    def reset(self):
        default_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_headpose = default_pose.copy()

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to client messages.

        Args:
            key: The message type to bind to. Valid values are "START", "STOP", and "RESET".
            func: The function to call when the message is received. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def _get_raw_data(self) -> Any:
        """Get the latest tracking data from the OpenXR runtime.

        Returns:
            Dictionary with TrackingTarget enum keys (HAND_LEFT, HAND_RIGHT, HEAD) containing:
                - Left hand joint poses: Dictionary of 26 joints with position and orientation
                - Right hand joint poses: Dictionary of 26 joints with position and orientation
                - Head pose: Single 7-element array with position and orientation

        Each pose is represented as a 7-element array: [x, y, z, qw, qx, qy, qz]
        where the first 3 elements are position and the last 4 are quaternion orientation.
        """
        return {
            self.TrackingTarget.HAND_LEFT: self._calculate_joint_poses(
                XRCore.get_singleton().get_input_device("/user/hand/left"),
                self._previous_joint_poses_left,
            ),
            self.TrackingTarget.HAND_RIGHT: self._calculate_joint_poses(
                XRCore.get_singleton().get_input_device("/user/hand/right"),
                self._previous_joint_poses_right,
            ),
            self.TrackingTarget.HEAD: self._calculate_headpose(),
        }

    """
    Internal helpers.
    """

    def _calculate_joint_poses(
        self, hand_device: Any, previous_joint_poses: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Calculate and update joint poses for a hand device.

        This function retrieves the current joint poses from the OpenXR hand device and updates
        the previous joint poses with the new data. If a joint's position or orientation is not
        valid, it will use the previous values.

        Args:
            hand_device: The OpenXR input device for a hand (/user/hand/left or /user/hand/right).
            previous_joint_poses: Dictionary mapping joint names to their previous poses.
                Each pose is a 7-element array: [x, y, z, qw, qx, qy, qz].

        Returns:
            Updated dictionary of joint poses with the same structure as previous_joint_poses.
            Each pose is represented as a 7-element numpy array: [x, y, z, qw, qx, qy, qz]
            where the first 3 elements are position and the last 4 are quaternion orientation.
        """
        if hand_device is None:
            return previous_joint_poses

        joint_poses = hand_device.get_all_virtual_world_poses()

        # Update each joint that is present in the current data
        for joint_name, joint_pose in joint_poses.items():
            if joint_name in HAND_JOINT_NAMES:
                # Extract translation and rotation
                if joint_pose.validity_flags & XRPoseValidityFlags.POSITION_VALID:
                    position = joint_pose.pose_matrix.ExtractTranslation()
                else:
                    position = previous_joint_poses[joint_name][:3]

                if joint_pose.validity_flags & XRPoseValidityFlags.ORIENTATION_VALID:
                    quat = joint_pose.pose_matrix.ExtractRotationQuat()
                    quati = quat.GetImaginary()
                    quatw = quat.GetReal()
                else:
                    quatw = previous_joint_poses[joint_name][3]
                    quati = previous_joint_poses[joint_name][4:]

                # Directly update the dictionary with new data
                previous_joint_poses[joint_name] = np.array(
                    [position[0], position[1], position[2], quatw, quati[0], quati[1], quati[2]], dtype=np.float32
                )

        # No need for conversion, just return the updated dictionary
        return previous_joint_poses

    def _calculate_headpose(self) -> np.ndarray:
        """Calculate the head pose from OpenXR.

        Returns:
            numpy.ndarray: 7-element array containing head position (xyz) and orientation (wxyz)
        """
        head_device = XRCore.get_singleton().get_input_device("/user/head")
        if head_device:
            hmd = head_device.get_virtual_world_pose("")
            position = hmd.ExtractTranslation()
            quat = hmd.ExtractRotationQuat()
            quati = quat.GetImaginary()
            quatw = quat.GetReal()

            # Store in w, x, y, z order to match our convention
            self._previous_headpose = np.array([
                position[0],
                position[1],
                position[2],
                quatw,
                quati[0],
                quati[1],
                quati[2],
            ])

        return self._previous_headpose

    def _on_teleop_command(self, event: carb.events.IEvent):
        msg = event.payload["message"]

        if "start" in msg:
            if "START" in self._additional_callbacks:
                self._additional_callbacks["START"]()
        elif "stop" in msg:
            if "STOP" in self._additional_callbacks:
                self._additional_callbacks["STOP"]()
        elif "reset" in msg:
            if "RESET" in self._additional_callbacks:
                self._additional_callbacks["RESET"]()
