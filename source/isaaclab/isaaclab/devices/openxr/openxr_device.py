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

    The device can track either the left hand, right hand, or both hands simultaneously based on
    the Hand enum value passed to the constructor. When retargeters are provided, the raw joint
    poses are transformed into robot control commands suitable for teleoperation.
    """

    class Hand(Enum):
        """Enum class specifying which hand(s) to track with OpenXR.

        Attributes:
            LEFT: Track only the left hand
            RIGHT: Track only the right hand
            BOTH: Track both hands simultaneously
        """

        LEFT = 0
        RIGHT = 1
        BOTH = 2

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"

    def __init__(
        self,
        xr_cfg: XrCfg | None,
        hand: Hand,
        retargeters: list[RetargeterBase] | None = None,
    ):
        """Initialize the hand tracking device.

        Args:
            xr_cfg: Configuration object for OpenXR settings. If None, default settings are used.
            hand: Which hand(s) to track (LEFT, RIGHT, or BOTH)
            retargeters: List of retargeters to transform hand tracking data into robot commands.
                        If None or empty list, raw joint poses will be returned.
        """
        super().__init__(retargeters)
        self._openxr = OpenXR()
        self._xr_cfg = xr_cfg or XrCfg()
        self._hand = hand
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

        # Specify the placement of the simulation when viewed in an XR device using a prim.
        xr_anchor = SingleXFormPrim("/XRAnchor", position=self._xr_cfg.anchor_pos, orientation=self._xr_cfg.anchor_rot)
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
        hand_str = "Both Hands" if self._hand == self.Hand.BOTH else f"{self._hand.name.title()} Hand"

        msg = f"OpenXR Hand Tracking Device: {self.__class__.__name__}\n"
        msg += f"\tTracking: {hand_str}\n"
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

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to client messages.

        Args:
            key: The message type to bind to. Valid values are "START", "STOP", and "RESET".
            func: The function to call when the message is received. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def _get_raw_data(self) -> Any:
        """Get the latest hand tracking data.

        Returns:
            Dictionary of joint poses
        """
        if self._hand == self.Hand.LEFT:
            hand_joints = self._openxr.locate_hand_joints(OpenXRSpec.XrHandEXT.XR_HAND_LEFT_EXT)
            return self._calculate_joint_poses(hand_joints, self._previous_joint_poses_left)
        elif self._hand == self.Hand.RIGHT:
            hand_joints = self._openxr.locate_hand_joints(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT)
            return self._calculate_joint_poses(hand_joints, self._previous_joint_poses_right)
        else:
            return {
                self.Hand.LEFT: self._calculate_joint_poses(
                    self._openxr.locate_hand_joints(OpenXRSpec.XrHandEXT.XR_HAND_LEFT_EXT),
                    self._previous_joint_poses_left,
                ),
                self.Hand.RIGHT: self._calculate_joint_poses(
                    self._openxr.locate_hand_joints(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT),
                    self._previous_joint_poses_right,
                ),
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
