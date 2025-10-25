# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction with motion controllers."""

import contextlib
import math
import numpy as np
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import carb
import omni.log
import usdrt
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf as pxrGf
from usdrt import Rt

import isaaclab.sim as sim_utils
from isaaclab.devices.retargeter_base import RetargeterBase

from .openxr_device import OpenXRDevice, OpenXRDeviceCfg
from .xr_cfg import XrAnchorRotationMode

with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore

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
    PADDING = 6  # Additional padding to make 7 elements to align with MotionControllerDataRowIndex.POSE


class MotionControllerDataRowIndex(Enum):
    """Enum for Motion Controller data row indices."""

    POSE = 0  # [x, y, z, w, x, y, z] - position and quaternion
    INPUTS = 1  # MotionControllerInputIndex: [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1]


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

        self._xr_core = XRCore.get_singleton()
        if self._xr_core is None:
            raise RuntimeError("XRCore is not available")

        # Force FOLLOW_PRIM_SMOOTHED mode for motion controller devices.
        if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FIXED:
            self._xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None  # For FOLLOW_PRIM_SMOOTHED mode
        self.__last_smoothing_update_time = None  # For wall clock time tracking
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True

        stage = get_current_stage()
        xr_anchor_headset_prim = stage.GetPrimAtPath(self._xr_anchor_headset_path)
        # Get the layer where it was created
        prim_stack = xr_anchor_headset_prim.GetPrimStack() if xr_anchor_headset_prim is not None else None
        if prim_stack:
            self.__anchor_headset_layer_identifier = prim_stack[0].layer.identifier
        else:
            self.__anchor_headset_layer_identifier = None
            omni.log.warn(f"Failed to get layer identifier for {self._xr_anchor_headset_path}")

        # Dictionary to store button event subscriptions
        # Key format: "device_path/button_name"
        # Value: dict with "generator", "press_sub", "inputs_change_sub", "disable_sub"
        self.__button_subscriptions: dict[str, dict] = {}

        # When the right controller"a" button is pressed, toggle the anchor rotation.
        self._bind_button_press(
            "/user/hand/right",
            "a",
            "isaaclab_right_a",
            lambda ev: self._toggle_anchor_rotation(),
        )

    """
    Operations
    """

    def reset(self):
        super().reset()
        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_smoothing_update_time = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self._sync_headset_to_anchor()

    def on_pre_render(self) -> None:
        """Sync the headset to the anchor before rendering."""
        super().on_pre_render()
        self._sync_headset_to_anchor()

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
                MotionControllerTrackingTarget.LEFT, self._xr_core.get_input_device("/user/hand/left")
            ),
            MotionControllerTrackingTarget.RIGHT: self._query_controller(
                MotionControllerTrackingTarget.RIGHT, self._xr_core.get_input_device("/user/hand/right")
            ),
            OpenXRDevice.TrackingTarget.HEAD: self._calculate_headpose(),
        }

    """
    Internal helpers.
    """

    def _query_controller(self, tracking_target: MotionControllerTrackingTarget, input_device) -> np.array:
        """Calculate and update input device data"""

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
            position[0],
            position[1],
            position[2],  # x, y, z position
            quat.GetReal(),
            quat.GetImaginary()[0],
            quat.GetImaginary()[1],
            quat.GetImaginary()[2],  # w, x, y, z quaternion
        ]

        # Second row: controller input values (6 values + 1 padding)
        input_row = [
            thumbstick_x,  # MotionControllerInputIndex.THUMBSTICK_X
            thumbstick_y,  # MotionControllerInputIndex.THUMBSTICK_Y
            trigger,  # MotionControllerInputIndex.TRIGGER
            squeeze,  # MotionControllerInputIndex.SQUEEZE
            button_0,  # MotionControllerInputIndex.BUTTON_0
            button_1,  # MotionControllerInputIndex.BUTTON_1
            0.0,  # MotionControllerInputIndex.PADDING
        ]

        # Combine into 2D array: [pose(7), inputs(7)]
        return np.array([pose_row, input_row], dtype=np.float32)

    def _sync_headset_to_anchor(self):
        """Sync the headset to the anchor.
        The prim is assumed to be in Fabric/usdrt.
        The XR system can currently only handle anchors in usd.
        Therefore, we need to query the anchor_prim's transform from usdrt and set it to the XR anchor in usd.
        """
        if self._xr_cfg.anchor_prim_path is None:
            return

        stage_id = sim_utils.get_current_stage_id()
        rt_stage = usdrt.Usd.Stage.Attach(stage_id)
        if rt_stage is None:
            return

        rt_prim = rt_stage.GetPrimAtPath(self._xr_cfg.anchor_prim_path)
        if rt_prim is None:
            return

        rt_xformable = Rt.Xformable(rt_prim)
        if rt_xformable is None:
            return

        world_matrix_attr = rt_xformable.GetFabricHierarchyWorldMatrixAttr()
        if world_matrix_attr is None:
            return

        rt_matrix = world_matrix_attr.Get()
        rt_pos = rt_matrix.ExtractTranslation()

        if self.__anchor_prim_initial_quat is None:
            self.__anchor_prim_initial_quat = rt_matrix.ExtractRotationQuat()

        # For comfort, we keep the anchor at a constant height instead of bobbing up and down when following robot joint.
        # We will need to update this value when robot crouches/stands.
        if self.__anchor_prim_initial_height is None:
            self.__anchor_prim_initial_height = rt_pos[2]

        rt_pos[2] = self.__anchor_prim_initial_height

        pxr_anchor_pos = pxrGf.Vec3d(*rt_pos) + pxrGf.Vec3d(*self._xr_cfg.anchor_pos)

        w, x, y, z = self._xr_cfg.anchor_rot
        pxr_cfg_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

        # XrAnchorRotationMode.FIXED is implicitly handled by setting pxr_anchor_quat to pxr_cfg_quat and overwritten in other modesZ
        pxr_anchor_quat = pxr_cfg_quat

        # XrAnchorRotationMode.FOLLOW_PRIM or XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        if (
            self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM
            or self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        ):
            # Calculate the delta rotation between the prim and the initial rotation
            rt_prim_quat = rt_matrix.ExtractRotationQuat()
            rt_delta_quat = rt_prim_quat * self.__anchor_prim_initial_quat.GetInverse()
            # Convert from usdrt to pxr quaternion
            pxr_delta_quat = pxrGf.Quatd(rt_delta_quat.GetReal(), pxrGf.Vec3d(*rt_delta_quat.GetImaginary()))

            # Only keep yaw rotation (assumes Z up axis):
            w = pxr_delta_quat.GetReal()
            ix, iy, iz = pxr_delta_quat.GetImaginary()

            # yaw around Z (right-handed, Z-up)
            yaw = math.atan2(2.0 * (w * iz + ix * iy), 1.0 - 2.0 * (iy * iy + iz * iz))

            # yaw-only quaternion about Z
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            pxr_delta_yaw_only_quat = pxrGf.Quatd(cy, pxrGf.Vec3d(0.0, 0.0, sy))
            pxr_anchor_quat = pxr_delta_yaw_only_quat * pxr_cfg_quat

            if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                # Initialize smoothed quaternion on first run
                if self.__smoothed_anchor_quat is None:
                    self.__smoothed_anchor_quat = pxr_anchor_quat
                    self.__last_smoothing_update_time = time.time()
                else:
                    # Calculate smoothing alpha from wall-clock time delta (not physics dt)
                    # Exponential smoothing: alpha = 1 - exp(-dt / time_constant)
                    current_time = time.time()
                    if self.__last_smoothing_update_time is None:
                        # Fallback in case reset didn't happen properly
                        self.__last_smoothing_update_time = current_time
                    dt = current_time - self.__last_smoothing_update_time
                    self.__last_smoothing_update_time = current_time

                    # Allow very small alpha for strong smoothing; only clamp upper bound
                    alpha = 1.0 - math.exp(-dt / max(self._xr_cfg.anchor_rotation_smoothing_time, 1e-6))
                    alpha = min(1.0, max(0.05, alpha))  # small floor avoids lingering

                    # Perform spherical linear interpolation (slerp)
                    # Use Gf.Slerp(alpha, quat_from, quat_to)
                    self.__smoothed_anchor_quat = pxrGf.Slerp(alpha, self.__smoothed_anchor_quat, pxr_anchor_quat)
                    pxr_anchor_quat = self.__smoothed_anchor_quat

        # XrAnchorRotationMode.CUSTOM
        elif self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM:
            if self._xr_cfg.anchor_rotation_custom_func is not None:
                rt_prim_quat = rt_matrix.ExtractRotationQuat()

                anchor_prim_pose = np.array(
                    [
                        rt_pos[0],
                        rt_pos[1],
                        rt_pos[2],
                        rt_prim_quat.GetReal(),
                        rt_prim_quat.GetImaginary()[0],
                        rt_prim_quat.GetImaginary()[1],
                        rt_prim_quat.GetImaginary()[2],
                    ],
                    dtype=np.float64,
                )
                np_array_quat = self._xr_cfg.anchor_rotation_custom_func(self._previous_headpose, anchor_prim_pose)

                w, x, y, z = np_array_quat
                pxr_anchor_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))
            else:
                print("[WARNING]: Anchor rotation custom function is not set. Using default rotation.")

        # Create the final matrix with combined rotation and adjusted position
        pxr_mat = pxrGf.Matrix4d()
        pxr_mat.SetTranslateOnly(pxr_anchor_pos)

        if self.__anchor_rotation_enabled:
            pxr_mat.SetRotateOnly(pxr_anchor_quat)
            self.__last_anchor_quat = pxr_anchor_quat
        else:
            pxr_mat.SetRotateOnly(self.__last_anchor_quat)
            self.__smoothed_anchor_quat = self.__last_anchor_quat

        self._xr_core.set_world_transform_matrix(
            self._xr_anchor_headset_path, pxr_mat, self.__anchor_headset_layer_identifier
        )

    def _bind_button_press(
        self,
        device_path: str,
        button_name: str,
        event_name: str,
        on_button_press: Callable[[carb.events.IEvent], None],
    ) -> None:
        """Bind a callback to a button press event on a motion controller.

        This method sets up event handling for a specific button on an XR controller.
        The callback will be invoked whenever the button is pressed. The binding
        automatically handles controller connection/disconnection and attempts to
        rebind when the device becomes available.

        Args:
            device_path: Path to the XR input device (e.g., "/user/hand/left" or "/user/hand/right")
            button_name: Name of the button to bind (e.g., "a", "b", "x", "y", "menu", "trigger", "squeeze")
            event_name: Unique event name for this binding (used internally for event routing)
            on_button_press: Callback function to invoke when the button is pressed.
                           The callback receives a carb.events.IEvent parameter.

        Example:
            >>> def on_menu_press(event):
            ...     print("Menu button pressed!")
            >>> device.bind_button_press("/user/hand/right", "menu", "toggle_menu", on_menu_press)
        """
        if XRCore is None:
            omni.log.warn("XR core not available; skipping button binding")
            return

        sub_key = f"{device_path}/{button_name}"
        self.__button_subscriptions[sub_key] = {}

        # Buttons don't automatically emit events when pressed, we need to bind an event generator to the button.
        # The device may be a stub until the real controller is connected so this may get called multiple times
        def try_emit_button_events():
            # Skip if already bound to avoid churn on repeated inputs_change
            if self.__button_subscriptions[sub_key].get("generator"):
                return

            device = self._xr_core.get_input_device(device_path)
            if not device:
                return

            # Only bind once the real controller exposes the button_name input
            names = {str(n) for n in (device.get_input_names() or ())}
            if button_name not in names:
                return

            gen = device.bind_event_generator(button_name, event_name, ("press",))
            if gen is not None:
                omni.log.info(f"XR: Bound event generator for {sub_key}, {event_name}")
                self.__button_subscriptions[sub_key]["generator"] = gen

        def on_inputs_change(_ev: carb.events.IEvent) -> None:
            try_emit_button_events()

        def on_disable(_ev: carb.events.IEvent) -> None:
            self.__button_subscriptions[sub_key]["generator"] = None

        message_bus = self._xr_core.get_message_bus()

        # Convert "/user/hand/right" -> "user_hand_right" for XR input event topics
        event_suffix = device_path.strip("/").replace("/", "_")

        # Subscribe to emitted button event
        self.__button_subscriptions[sub_key]["press_sub"] = message_bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(f"{event_name}.press"), on_button_press
        )

        # Re-attempt binding when inputs change (real controller arrives)
        self.__button_subscriptions[sub_key]["inputs_change_sub"] = message_bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(f"xr_input.{event_suffix}.inputs_change"), on_inputs_change
        )

        # Handle controller disconnection
        self.__button_subscriptions[sub_key]["disable_sub"] = message_bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(f"xr_input.{event_suffix}.disable"), on_disable
        )

        # Attempt initial binding
        try_emit_button_events()

    def _unbind_all_buttons(self) -> None:
        """Unbind all button press events.

        This cleans up all button event subscriptions and generators. Typically called
        during device cleanup or reset.
        """
        for sub_key, subs in self.__button_subscriptions.items():
            # Clean up generator
            if "generator" in subs:
                subs["generator"] = None

            # Clean up subscriptions
            for key in ["press_sub", "inputs_change_sub", "disable_sub"]:
                if key in subs:
                    subs[key] = None

        self.__button_subscriptions.clear()
        omni.log.info("XR: Unbound all button event handlers")

    def _toggle_anchor_rotation(self):
        """Toggle the anchor rotation."""
        self.__anchor_rotation_enabled = not self.__anchor_rotation_enabled
        omni.log.info(f"Toggling anchor rotation: {self.__anchor_rotation_enabled}")
