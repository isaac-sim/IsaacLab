# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manus and Vive for teleoperation and interaction.

.. deprecated::
    :class:`ManusVive` and :class:`ManusViveCfg` are deprecated.
    Please use :class:`isaaclab_teleop.IsaacTeleopDevice` and
    :class:`isaaclab_teleop.IsaacTeleopCfg` instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from packaging import version

import carb

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.utils.version import get_isaac_sim_version

from .common import HAND_JOINT_NAMES
from .xr_cfg import XrCfg

# For testing purposes, we need to mock the XRCore
XRCore = None

with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore

import isaaclab.sim as sim_utils

from .manus_vive_utils import HAND_JOINT_MAP, ManusViveIntegration


class ManusVive(DeviceBase):
    """Manus gloves and Vive trackers for teleoperation and interaction.

    This device tracks hand joints using Manus gloves and Vive trackers and makes them available as:

    1. A dictionary of tracking data (when used without retargeters)
    2. Retargeted commands for robot control (when retargeters are provided)

    The user needs to install the Manus SDK and add `{path_to_manus_sdk}/manus_sdk/lib` to `LD_LIBRARY_PATH`.
    Data are acquired by `ManusViveIntegration` from `isaaclab_teleop.deprecated.openxr.manus_vive_utils`, including

    * Vive tracker poses in scene frame, calibrated from AVP wrist poses.
    * Hand joints calculated from Vive wrist joints and Manus hand joints (relative to wrist).
    * Vive trackers are automatically mapped to the left and right wrist joints.

    Raw data format (_get_raw_data output): consistent with :class:`OpenXRDevice`.
    Joint names are defined in `HAND_JOINT_MAP` from `isaaclab_teleop.deprecated.openxr.manus_vive_utils`.

    Teleop commands: consistent with :class:`OpenXRDevice`.

    The device tracks the left hand, right hand, head position, or any combination of these
    based on the TrackingTarget enum values. When retargeters are provided, the raw tracking
    data is transformed into robot control commands suitable for teleoperation.
    """

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"

    def __init__(self, cfg: ManusViveCfg, retargeters: list[RetargeterBase] | None = None):
        """Initialize the Manus+Vive device.

        Args:
            cfg: Configuration object for Manus+Vive settings.
            retargeters: List of retargeter instances to use for transforming raw tracking data.
        """
        import warnings

        warnings.warn(
            "ManusVive is deprecated. Please use isaaclab_teleop.IsaacTeleopDevice instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(retargeters)
        # Enforce minimum Isaac Sim version (>= 5.1)
        isaac_sim_version = get_isaac_sim_version()
        if isaac_sim_version < version.parse("5.1"):
            raise RuntimeError(f"ManusVive requires Isaac Sim >= 5.1. Detected version: '{isaac_sim_version}'.")
        self._xr_cfg = cfg.xr_cfg or XrCfg()
        self._additional_callbacks = dict()
        self._vc_subscription = (
            XRCore.get_singleton()
            .get_message_bus()
            .create_subscription_to_pop_by_type(
                carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
            )
        )
        self._manus_vive = ManusViveIntegration()

        # Initialize dictionaries instead of arrays
        default_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_headpose = default_pose.copy()

        xr_anchor_prim_path = "/XRAnchor"
        sim_utils.create_prim(
            xr_anchor_prim_path,
            prim_type="Xform",
            position=self._xr_cfg.anchor_pos,
            orientation=self._xr_cfg.anchor_rot,
        )
        carb.settings.get_settings().set_float("/persistent/xr/render/nearPlane", self._xr_cfg.near_plane)
        carb.settings.get_settings().set_string("/persistent/xr/anchorMode", "custom anchor")
        carb.settings.get_settings().set_string("/xrstage/customAnchor", xr_anchor_prim_path)

    def __del__(self):
        """Clean up resources when the object is destroyed.
        Properly unsubscribes from the XR message bus to prevent memory leaks
        and resource issues when the device is no longer needed.
        """
        if hasattr(self, "_vc_subscription") and self._vc_subscription is not None:
            self._vc_subscription = None

        # No need to explicitly clean up OpenXR instance as it's managed by NVIDIA Isaac Sim

    def __str__(self) -> str:
        """Provide details about the device configuration, tracking settings,
        and available gesture commands.

        Returns:
            Formatted string with device information.
        """

        msg = f"Manus+Vive Hand Tracking Device: {self.__class__.__name__}\n"
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
        msg += "\tTracked Joints: 26 XR hand joints including:\n"
        msg += "\t\t- Wrist, palm\n"
        msg += "\t\t- Thumb (tip, intermediate joints)\n"
        msg += "\t\t- Fingers (tip, distal, intermediate, proximal)\n"

        return msg

    def reset(self):
        """Reset cached joint and head poses."""
        default_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_headpose = default_pose.copy()

    def add_callback(self, key: str, func: Callable):
        """Register a callback for a given key.

        Args:
            key: The message key to bind ('START', 'STOP', 'RESET').
            func: The function to invoke when the message key is received.
        """
        self._additional_callbacks[key] = func

    def _get_raw_data(self) -> dict:
        """Get the latest tracking data from Manus and Vive.

        Returns:
            Dictionary with TrackingTarget enum keys (HAND_LEFT, HAND_RIGHT, HEAD) containing:
                - Left hand joint poses: Dictionary of 26 joints with position and orientation
                - Right hand joint poses: Dictionary of 26 joints with position and orientation
                - Head pose: Single 7-element array with position and orientation

        Each pose is represented as a 7-element array: [x, y, z, qx, qy, qz, qw]
        where the first 3 elements are position and the last 4 are quaternion orientation.
        """
        hand_tracking_data = self._manus_vive.get_all_device_data()["manus_gloves"]
        result = {"left": self._previous_joint_poses_left, "right": self._previous_joint_poses_right}
        for joint, pose in hand_tracking_data.items():
            hand, index = joint.split("_")
            joint_name = HAND_JOINT_MAP[int(index)]
            result[hand][joint_name] = np.array(pose["position"] + pose["orientation"], dtype=np.float32)
        return {
            DeviceBase.TrackingTarget.HAND_LEFT: result["left"],
            DeviceBase.TrackingTarget.HAND_RIGHT: result["right"],
            DeviceBase.TrackingTarget.HEAD: self._calculate_headpose(),
        }

    def _calculate_headpose(self) -> np.ndarray:
        """Calculate the head pose from OpenXR.

        Returns:
            7-element numpy.ndarray [x, y, z, qx, qy, qz, qw].
        """
        head_device = XRCore.get_singleton().get_input_device("/user/head")
        if head_device:
            hmd = head_device.get_virtual_world_pose("")
            position = hmd.ExtractTranslation()
            quat = hmd.ExtractRotationQuat()
            quati = quat.GetImaginary()
            quatw = quat.GetReal()

            # Store in x, y, z, w order to match our convention
            self._previous_headpose = np.array(
                [
                    position[0],
                    position[1],
                    position[2],
                    quati[0],
                    quati[1],
                    quati[2],
                    quatw,
                ]
            )

        return self._previous_headpose

    def _on_teleop_command(self, event: carb.events.IEvent):
        """Handle teleoperation command events.

        Args:
            event: The XR message-bus event containing a 'message' payload.
        """
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


@dataclass
class ManusViveCfg(DeviceCfg):
    """Configuration for Manus and Vive."""

    xr_cfg: XrCfg | None = None
    class_type: type[DeviceBase] = ManusVive
