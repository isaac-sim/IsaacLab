# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction.

.. deprecated::
    :class:`OpenXRDevice` and :class:`OpenXRDeviceCfg` are deprecated.
    Please use :class:`isaaclab_teleop.IsaacTeleopDevice` and
    :class:`isaaclab_teleop.IsaacTeleopCfg` instead.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import carb

# import logger
logger = logging.getLogger(__name__)

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase

from .common import HAND_JOINT_NAMES
from .xr_anchor_utils import XrAnchorSynchronizer
from .xr_cfg import XrCfg

# For testing purposes, we need to mock the XRCore, XRPoseValidityFlags classes
XRCore = None
XRPoseValidityFlags = None
XRCoreEventType = None

with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRCoreEventType, XRPoseValidityFlags

import isaaclab.sim as sim_utils


class OpenXRDevice(DeviceBase):
    """An OpenXR-powered device for teleoperation and interaction.

    This device tracks hand joints using OpenXR and makes them available as:

    1. A dictionary of tracking data (when used without retargeters)
    2. Retargeted commands for robot control (when retargeters are provided)

    Raw data format (_get_raw_data output):

    * A dictionary with keys matching TrackingTarget enum values (HAND_LEFT, HAND_RIGHT, HEAD)
    * Each hand tracking entry contains a dictionary of joint poses
    * Each joint pose is a 7D vector (x, y, z, qw, qx, qy, qz) in meters and quaternion units
    * Joint names are defined in HAND_JOINT_NAMES from isaaclab_teleop.deprecated.openxr.common
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
        import warnings

        warnings.warn(
            "OpenXRDevice is deprecated. Please use isaaclab_teleop.IsaacTeleopDevice instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(retargeters)
        self._xr_cfg = cfg.xr_cfg or XrCfg()
        self._additional_callbacks = dict()
        self._xr_core = XRCore.get_singleton() if XRCore is not None else None
        self._vc_subscription = (
            self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
                carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
            )
            if self._xr_core is not None
            else None
        )

        # Initialize dictionaries instead of arrays
        default_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_headpose = default_pose.copy()

        if self._xr_cfg.anchor_prim_path is not None:
            anchor_path = self._xr_cfg.anchor_prim_path
            if anchor_path.endswith("/"):
                anchor_path = anchor_path[:-1]
            self._xr_anchor_headset_path = f"{anchor_path}/XRAnchor"
        else:
            self._xr_anchor_headset_path = "/World/XRAnchor"

        # Only create the anchor prim if it doesn't already exist (supports multiple devices sharing anchor)
        stage = sim_utils.get_current_stage()
        if not stage.GetPrimAtPath(self._xr_anchor_headset_path).IsValid():
            sim_utils.create_prim(
                self._xr_anchor_headset_path,
                prim_type="Xform",
                position=self._xr_cfg.anchor_pos,
                orientation=self._xr_cfg.anchor_rot,
            )

        if hasattr(carb, "settings"):
            carb.settings.get_settings().set_float("/persistent/xr/render/nearPlane", self._xr_cfg.near_plane)
            carb.settings.get_settings().set_string("/persistent/xr/anchorMode", "custom anchor")
            carb.settings.get_settings().set_string("/xrstage/customAnchor", self._xr_anchor_headset_path)

        # Button binding support
        self.__button_subscriptions: dict[str, dict] = {}

        # Optional anchor synchronizer
        self._anchor_sync: XrAnchorSynchronizer | None = None
        if self._xr_core is not None and self._xr_cfg.anchor_prim_path is not None:
            try:
                self._anchor_sync = XrAnchorSynchronizer(
                    xr_core=self._xr_core,
                    xr_cfg=self._xr_cfg,
                    xr_anchor_headset_path=self._xr_anchor_headset_path,
                )
                # Subscribe to pre_sync_update to keep anchor in sync
                if XRCoreEventType is not None:
                    self._xr_pre_sync_update_subscription = (
                        self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
                            XRCoreEventType.pre_sync_update,
                            lambda _: self._anchor_sync.sync_headset_to_anchor(),
                            name="isaaclab_xr_pre_sync_update",
                        )
                    )
            except Exception as e:
                logger.warning(f"XR: Failed to initialize anchor synchronizer: {e}")

        # Default convenience binding: toggle anchor rotation with right controller 'a' button
        with contextlib.suppress(Exception):
            self._bind_button_press(
                "/user/hand/right",
                "a",
                "isaaclab_right_a",
                lambda ev: self._toggle_anchor_rotation(),
            )

    def __del__(self):
        """Clean up resources when the object is destroyed.

        Properly unsubscribes from the XR message bus to prevent memory leaks
        and resource issues when the device is no longer needed.
        """
        if hasattr(self, "_vc_subscription") and self._vc_subscription is not None:
            self._vc_subscription = None
        if hasattr(self, "_xr_pre_sync_update_subscription") and self._xr_pre_sync_update_subscription is not None:
            self._xr_pre_sync_update_subscription = None
        # clear button subscriptions
        if hasattr(self, "__button_subscriptions"):
            self._unbind_all_buttons()

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
        if self._xr_cfg.anchor_prim_path is not None:
            msg += f"\tAnchor Prim Path: {self._xr_cfg.anchor_prim_path} (Dynamic Anchoring)\n"
        else:
            msg += "\tAnchor Mode: Static (Root Level)\n"

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
        if hasattr(self, "_anchor_sync") and self._anchor_sync is not None:
            self._anchor_sync.reset()

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
        data = {}

        if RetargeterBase.Requirement.HAND_TRACKING in self._required_features:
            data[DeviceBase.TrackingTarget.HAND_LEFT] = self._calculate_joint_poses(
                XRCore.get_singleton().get_input_device("/user/hand/left"),
                self._previous_joint_poses_left,
            )
            data[DeviceBase.TrackingTarget.HAND_RIGHT] = self._calculate_joint_poses(
                XRCore.get_singleton().get_input_device("/user/hand/right"),
                self._previous_joint_poses_right,
            )

        if RetargeterBase.Requirement.HEAD_TRACKING in self._required_features:
            data[DeviceBase.TrackingTarget.HEAD] = self._calculate_headpose()

        if RetargeterBase.Requirement.MOTION_CONTROLLER in self._required_features:
            # Optionally include motion controller pose+inputs if devices are available
            try:
                left_dev = XRCore.get_singleton().get_input_device("/user/hand/left")
                right_dev = XRCore.get_singleton().get_input_device("/user/hand/right")
                left_ctrl = self._query_controller(left_dev) if left_dev is not None else np.array([])
                right_ctrl = self._query_controller(right_dev) if right_dev is not None else np.array([])
                if left_ctrl.size:
                    data[DeviceBase.TrackingTarget.CONTROLLER_LEFT] = left_ctrl
                if right_ctrl.size:
                    data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] = right_ctrl
            except Exception:
                # Ignore controller data if XRCore/controller APIs are unavailable
                pass

        return data

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
                    [position[0], position[1], position[2], quati[0], quati[1], quati[2], quatw], dtype=np.float32
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

    # -----------------------------
    # Controller button binding utilities
    # -----------------------------
    def _bind_button_press(
        self,
        device_path: str,
        button_name: str,
        event_name: str,
        on_button_press: Callable[[carb.events.IEvent], None],
    ) -> None:
        if self._xr_core is None:
            logger.warning("XR core not available; skipping button binding")
            return

        sub_key = f"{device_path}/{button_name}"
        self.__button_subscriptions[sub_key] = {}

        def try_emit_button_events():
            if self.__button_subscriptions[sub_key].get("generator"):
                return
            device = self._xr_core.get_input_device(device_path)
            if not device:
                return
            names = {str(n) for n in (device.get_input_names() or ())}
            if button_name not in names:
                return
            gen = device.bind_event_generator(button_name, event_name, ("press",))
            if gen is not None:
                logger.info(f"XR: Bound event generator for {sub_key}, {event_name}")
                self.__button_subscriptions[sub_key]["generator"] = gen

        def on_inputs_change(_ev: carb.events.IEvent) -> None:
            try_emit_button_events()

        def on_disable(_ev: carb.events.IEvent) -> None:
            self.__button_subscriptions[sub_key]["generator"] = None

        message_bus = self._xr_core.get_message_bus()
        event_suffix = device_path.strip("/").replace("/", "_")
        self.__button_subscriptions[sub_key]["press_sub"] = message_bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(f"{event_name}.press"), on_button_press
        )
        self.__button_subscriptions[sub_key]["inputs_change_sub"] = message_bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(f"xr_input.{event_suffix}.inputs_change"), on_inputs_change
        )
        self.__button_subscriptions[sub_key]["disable_sub"] = message_bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(f"xr_input.{event_suffix}.disable"), on_disable
        )
        try_emit_button_events()

    def _unbind_all_buttons(self) -> None:
        for sub_key, subs in self.__button_subscriptions.items():
            if "generator" in subs:
                subs["generator"] = None
            for key in ["press_sub", "inputs_change_sub", "disable_sub"]:
                if key in subs:
                    subs[key] = None
        self.__button_subscriptions.clear()
        logger.info("XR: Unbound all button event handlers")

    def _toggle_anchor_rotation(self):
        if self._anchor_sync is not None:
            self._anchor_sync.toggle_anchor_rotation()

    def _query_controller(self, input_device) -> np.ndarray:
        """Query motion controller pose and inputs as a 2x7 array.

        Row 0 (POSE): [x, y, z, qx, qy, qz, qw]
        Row 1 (INPUTS): [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, padding]
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
            thumbstick_x = float(input_device.get_input_gesture_value("thumbstick", "x"))
        if input_device.has_input_gesture("thumbstick", "y"):
            thumbstick_y = float(input_device.get_input_gesture_value("thumbstick", "y"))
        if input_device.has_input_gesture("trigger", "value"):
            trigger = float(input_device.get_input_gesture_value("trigger", "value"))
        if input_device.has_input_gesture("squeeze", "value"):
            squeeze = float(input_device.get_input_gesture_value("squeeze", "value"))

        # Determine which button pair exists on this device
        if input_device.has_input_gesture("x", "click") or input_device.has_input_gesture("y", "click"):
            if input_device.has_input_gesture("x", "click"):
                button_0 = float(input_device.get_input_gesture_value("x", "click"))
            if input_device.has_input_gesture("y", "click"):
                button_1 = float(input_device.get_input_gesture_value("y", "click"))
        else:
            if input_device.has_input_gesture("a", "click"):
                button_0 = float(input_device.get_input_gesture_value("a", "click"))
            if input_device.has_input_gesture("b", "click"):
                button_1 = float(input_device.get_input_gesture_value("b", "click"))

        pose_row = [
            position[0],
            position[1],
            position[2],
            quat.GetImaginary()[0],
            quat.GetImaginary()[1],
            quat.GetImaginary()[2],
            quat.GetReal(),
        ]

        input_row = [
            thumbstick_x,
            thumbstick_y,
            trigger,
            squeeze,
            button_0,
            button_1,
            0.0,
        ]

        return np.array([pose_row, input_row], dtype=np.float32)

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
            self.reset()


@dataclass
class OpenXRDeviceCfg(DeviceCfg):
    """Configuration for OpenXR devices."""

    xr_cfg: XrCfg | None = None
    class_type: type[DeviceBase] = OpenXRDevice
