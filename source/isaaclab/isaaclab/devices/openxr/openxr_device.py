# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction."""

import math
import contextlib
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import carb
import usdrt
from pxr import Gf as pxrGf
from usdrt import Rt

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase

from ..device_base import DeviceBase, DeviceCfg
from .xr_cfg import XrCfg, XrAnchorRotationMode

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

        if self._xr_cfg.anchor_prim_path is not None:
            self.__xr_anchor_headset_path = f"{self._xr_cfg.anchor_prim_path}/XRAnchor"
        else:
            self.__xr_anchor_headset_path = "/World/XRAnchor"

        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None  # For FOLLOW_PRIM_SMOOTHED mode

        xr_anchor_headset = SingleXFormPrim(
            self.__xr_anchor_headset_path, position=self._xr_cfg.anchor_pos, orientation=self._xr_cfg.anchor_rot
        )

        # Get the layer where it was created
        prim_stack = xr_anchor_headset.prim.GetPrimStack()
        if prim_stack:
            self.__anchor_headset_layer_identifier = prim_stack[0].layer.identifier

        carb.settings.get_settings().set_float("/persistent/xr/profile/ar/render/nearPlane", self._xr_cfg.near_plane)
        carb.settings.get_settings().set_string("/persistent/xr/profile/ar/anchorMode", "custom anchor")
        carb.settings.get_settings().set_string("/xrstage/profile/ar/customAnchor", self.__xr_anchor_headset_path )

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
        if self._xr_cfg.anchor_prim_path is not None:
            msg += f"\tAnchor Prim Path: {self._xr_cfg.anchor_prim_path} (Dynamic Anchoring)\n"
        else:
            msg += f"\tAnchor Mode: Static (Root Level)\n"

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
        print("Resetting OpenXR device")
        default_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_headpose = default_pose.copy()

        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None

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

    def on_pre_render(self) -> None:
        """Sync the headset to the anchor before rendering."""
        self._sync_headset_to_anchor()

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
        if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM or self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
            # Calculate the delta rotation between the prim and the initial rotation
            rt_prim_quat = rt_matrix.ExtractRotationQuat()
            rt_delta_quat = rt_prim_quat * self.__anchor_prim_initial_quat.GetInverse()
            # Convert from usdrt to pxr quaternion
            pxr_delta_quat = pxrGf.Quatd(rt_delta_quat.GetReal(), pxrGf.Vec3d(*rt_delta_quat.GetImaginary()))

            # Only keep yaw rotation (assumes Z up axis):
            w = pxr_delta_quat.GetReal()
            ix, iy, iz = pxr_delta_quat.GetImaginary()

            # yaw around Z (right-handed, Z-up)
            yaw = math.atan2(2.0 * (w*iz + ix*iy), 1.0 - 2.0 * (iy*iy + iz*iz))

            # yaw-only quaternion about Z
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            pxr_delta_yaw_only_quat = pxrGf.Quatd(cy, pxrGf.Vec3d(0.0, 0.0, sy))
            pxr_anchor_quat = pxr_delta_yaw_only_quat * pxr_cfg_quat

            if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                # Initialize smoothed quaternion on first run
                if self.__smoothed_anchor_quat is None:
                    self.__smoothed_anchor_quat = pxr_anchor_quat
                else:
                    # Calculate smoothing alpha from time constant
                    # alpha = 1 - exp(-dt / time_constant)
                    # This gives exponential smoothing behavior
                    dt = SimulationContext.instance().get_physics_dt()
                    alpha = 1.0 - math.exp(-dt / max(self._xr_cfg.anchor_rotation_smoothing_time, 0.001))
                    alpha = max(0.01, min(1.0, alpha))  # Minimum 1% blend prevents very slow updates

                    # Perform spherical linear interpolation (slerp)
                    # Use Gf.Slerp(alpha, quat_from, quat_to)
                    self.__smoothed_anchor_quat = pxrGf.Slerp(alpha, self.__smoothed_anchor_quat, pxr_anchor_quat)
                    pxr_anchor_quat = self.__smoothed_anchor_quat

        # XrAnchorRotationMode.CUSTOM
        elif self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM:
            if self._xr_cfg.anchor_rotation_custom_func is not None:
                rt_prim_quat = rt_matrix.ExtractRotationQuat()

                anchor_prim_pose = np.array([rt_pos[0], rt_pos[1], rt_pos[2], rt_prim_quat.GetReal(), rt_prim_quat.GetImaginary()[0], rt_prim_quat.GetImaginary()[1], rt_prim_quat.GetImaginary()[2]], dtype=np.float64)
                np_array_quat = self._xr_cfg.anchor_rotation_custom_func(self._previous_headpose, anchor_prim_pose)

                w, x, y, z = np_array_quat
                pxr_anchor_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))
            else:
                print("[WARNING]: Anchor rotation custom function is not set. Using default rotation.")

        # Create the final matrix with combined rotation and adjusted position
        pxr_mat = pxrGf.Matrix4d()
        pxr_mat.SetRotateOnly(pxr_anchor_quat)
        pxr_mat.SetTranslateOnly(pxr_anchor_pos)        

        XRCore.get_singleton().set_world_transform_matrix(self.__xr_anchor_headset_path, pxr_mat, self.__anchor_headset_layer_identifier)
