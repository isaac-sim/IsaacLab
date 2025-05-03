# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction."""

import contextlib
import numpy as np
from collections.abc import Callable
from enum import Enum
from typing import Any

import carb

from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase

from ..device_base import DeviceBase
from .xr_cfg import XrCfg

with contextlib.suppress(ModuleNotFoundError):
    from isaacsim.xr.openxr import OpenXR, OpenXRSpec
    from omni.kit.xr.core import XRCore

from isaacsim.core.prims import SingleXFormPrim


class OpenXRDevice(DeviceBase):
    """An OpenXR-powered device for teleoperation and interaction.

    This device tracks hand joints using OpenXR and makes them available as:
    1. A dictionary of joint poses (when used directly)
    2. Retargeted commands for robot control (when a retargeter is provided)

    Data format:
    * Joint poses: Each pose is a 7D vector (x, y, z, qw, qx, qy, qz) in meters and quaternion units
    * Dictionary keys: Joint names from HAND_JOINT_NAMES in isaaclab.devices.openxr.common
    * Supported joints include palm, wrist, and joints for thumb, index, middle, ring and little fingers

    Teleop commands:
    The device responds to several teleop commands that can be subscribed to via add_callback():
    * "START": Resume hand tracking data flow
    * "STOP": Pause hand tracking data flow
    * "RESET": Reset the tracking and signal simulation reset

    The device can track the left hand, right hand, head position, or any combination of these
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
        xr_cfg: XrCfg | None,
        retargeters: list[RetargeterBase] | None = None,
    ):
        """Initialize the OpenXR device.

        Args:
            xr_cfg: Configuration object for OpenXR settings. If None, default settings are used.
            retargeters: List of retargeters to transform tracking data into robot commands.
                        If None or empty list, raw tracking data will be returned.
        """
        super().__init__(retargeters)
        self._openxr = OpenXR()
        self._xr_cfg = xr_cfg or XrCfg()
        self._additional_callbacks = dict()
        self._vc_subscription = (
            XRCore.get_singleton()
            .get_message_bus()
            .create_subscription_to_pop_by_type(
                carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
            )
        )
        self._previous_joint_poses_left = np.full((26, 7), [0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_right = np.full((26, 7), [0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_headpose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

        # Specify the placement of the simulation when viewed in an XR device using a prim.
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
        self._previous_joint_poses_left = np.full((26, 7), [0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_right = np.full((26, 7), [0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_headpose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

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
            Dictionary containing tracking data for:
                - Left hand joint poses (26 joints with position and orientation)
                - Right hand joint poses (26 joints with position and orientation)
                - Head pose (position and orientation)

        Each pose is represented as a 7-element array: [x, y, z, qw, qx, qy, qz]
        where the first 3 elements are position and the last 4 are quaternion orientation.
        """
        return {
            self.TrackingTarget.HAND_LEFT: self._calculate_joint_poses(
                self._openxr.locate_hand_joints(OpenXRSpec.XrHandEXT.XR_HAND_LEFT_EXT),
                self._previous_joint_poses_left,
            ),
            self.TrackingTarget.HAND_RIGHT: self._calculate_joint_poses(
                self._openxr.locate_hand_joints(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT),
                self._previous_joint_poses_right,
            ),
            self.TrackingTarget.HEAD: self._calculate_headpose(),
        }

    """
    Internal helpers.
    """

    def _calculate_joint_poses(self, hand_joints, previous_joint_poses) -> dict[str, np.ndarray]:
        if hand_joints is None:
            return self._joints_to_dict(previous_joint_poses)

        hand_joints = np.array(hand_joints)
        positions = np.array([[j.pose.position.x, j.pose.position.y, j.pose.position.z] for j in hand_joints])
        orientations = np.array([
            [j.pose.orientation.w, j.pose.orientation.x, j.pose.orientation.y, j.pose.orientation.z]
            for j in hand_joints
        ])
        location_flags = np.array([j.locationFlags for j in hand_joints])

        pos_mask = (location_flags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0
        ori_mask = (location_flags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0

        previous_joint_poses[pos_mask, 0:3] = positions[pos_mask]
        previous_joint_poses[ori_mask, 3:7] = orientations[ori_mask]

        return self._joints_to_dict(previous_joint_poses)

    def _calculate_headpose(self) -> np.ndarray:
        """Calculate the head pose from OpenXR.

        Returns:
            numpy.ndarray: 7-element array containing head position (xyz) and orientation (wxyz)
        """
        head_device = XRCore.get_singleton().get_input_device("displayDevice")
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

    def _joints_to_dict(self, joint_data: np.ndarray) -> dict[str, np.ndarray]:
        """Convert joint array to dictionary using standard joint names.

        Args:
            joint_data: Array of joint data (Nx6 for N joints)

        Returns:
            Dictionary mapping joint names to their data
        """
        return {joint_name: joint_data[i] for i, joint_name in enumerate(HAND_JOINT_NAMES)}

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
