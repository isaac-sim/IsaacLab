# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR handtracking controller for SE(3) control."""
import contextlib
import numpy as np
from collections.abc import Callable
from scipy.spatial.transform import Rotation, Slerp
from typing import Final

import carb

from ..device_base import DeviceBase
from .xr_cfg import XrCfg

with contextlib.suppress(ModuleNotFoundError):
    from isaacsim.xr.openxr import OpenXR, OpenXRSpec
    from omni.kit.xr.core import XRCore

    from . import teleop_command

from isaacsim.core.prims import SingleXFormPrim

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


class Se3HandTracking(DeviceBase):
    """A OpenXR powered hand tracker for sending SE(3) commands as delta poses
    as well as teleop commands via callbacks.

    The command comprises of two parts as well as callbacks for teleop commands:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians while in rel mode
    * or a 7D vector of (x, y, z, qw, qx, qy, qz) (meters/quat) in abs mode.
    * The pose is based on the center of the "pinch" gesture, aka the average position/rotation of the
    * thumb tip and index finger tip described here:
    * https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints
    * If use_wrist_position is True, the wrist position is used instead, this is the default behavior.
    * Note: delta_pos_scale_factor and delta_rot_scale_factor do not apply in abs mode.
    * gripper: a binary command to open or close the gripper.
    *
    * Teleop commands can be subscribed via the add_callback function, the available keys are:
    * - "START": Resumes hand tracking flow
    * - "STOP": Stops hand tracking flow
    * - "RESET": signals simulation reset
    """

    GRIP_HYSTERESIS_METERS: Final[float] = 0.05  # 2.0 inch

    def __init__(
        self,
        xr_cfg: XrCfg | None,
        hand,
        abs=True,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=True,
        delta_pos_scale_factor=10,
        delta_rot_scale_factor=10,
    ):
        self._openxr = OpenXR()
        self._previous_pos = np.zeros(3)
        self._previous_rot = Rotation.identity()
        self._previous_grip_distance = 0.0
        self._previous_gripper_command = False
        self._xr_cfg = xr_cfg or XrCfg()
        self._hand = hand
        self._abs = abs
        self._zero_out_xy_rotation = zero_out_xy_rotation
        self._use_wrist_rotation = use_wrist_rotation
        self._use_wrist_position = use_wrist_position
        self._additional_callbacks = dict()
        self._vc = teleop_command.TeleopCommand()
        self._vc_subscription = (
            XRCore.get_singleton()
            .get_message_bus()
            .create_subscription_to_pop_by_type(
                carb.events.type_from_string(teleop_command.TELEOP_COMMAND_RESPONSE_EVENT_TYPE), self._on_teleop_command
            )
        )
        self._tracking = False

        self._alpha_pos = 0.5
        self._alpha_rot = 0.5
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)
        self._delta_pos_scale_factor = delta_pos_scale_factor
        self._delta_rot_scale_factor = delta_rot_scale_factor
        self._frame_marker_cfg = FRAME_MARKER_CFG.copy()
        self._frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self._goal_marker = VisualizationMarkers(self._frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # Specify the placement of the simulation when viewed in an XR device using a prim.
        xr_anchor = SingleXFormPrim("/XRAnchor", position=self._xr_cfg.anchor_pos, orientation=self._xr_cfg.anchor_rot)
        carb.settings.get_settings().set_string("/persistent/xr/profile/ar/anchorMode", "custom anchor")
        carb.settings.get_settings().set_string("/xrstage/profile/ar/customAnchor", xr_anchor.prim_path)

    def __del__(self):
        return

    def __str__(self) -> str:
        return ""

    """
    Operations
    """

    def reset(self):
        self._previous_pos = np.zeros(3)
        self._previous_rot = Rotation.identity()
        self._previous_grip_distance = 0.0
        self._previous_gripper_command = False

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from teleop device event state.

        Returns:
            A tuple containing the delta or abs pose command and gripper commands.
        """

        hand_joints = self._openxr.locate_hand_joints(self._hand)

        return self._calculate_target_pose(hand_joints), self._calculate_gripper_command(hand_joints)

    def visualize(self, enable):
        if enable:
            self._goal_marker.set_visibility(True)
            trans = np.array([self._previous_pos])
            quat = self._previous_rot.as_quat()
            rot = np.array([np.array([quat[3], quat[0], quat[1], quat[2]])])
            self._goal_marker.visualize(translations=trans, orientations=rot)
        else:
            self._goal_marker.set_visibility(False)

    """
    Internal helpers.
    """

    def _calculate_target_pose(self, hand_joints):
        if self._abs:
            return self._calculate_abs_target_pose(hand_joints)
        else:
            return self._calculate_target_delta_pose(hand_joints)

    def _calculate_abs_target_pose(self, hand_joints):
        if hand_joints is None or not self._tracking:
            previous_rot = self._previous_rot.as_quat()
            previous_rot = [previous_rot[3], previous_rot[0], previous_rot[1], previous_rot[2]]
            return np.concatenate([self._previous_pos, previous_rot])

        index_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_INDEX_TIP_EXT]
        thumb_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_THUMB_TIP_EXT]
        wrist = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_WRIST_EXT]

        # position:
        if self._use_wrist_position:
            if not wrist.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT:
                target_pos = self._previous_pos
            else:
                wrist_pos = wrist.pose.position
                wrist_position = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z], dtype=np.float32)
                target_pos = wrist_position

        else:
            if (
                not index_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT
                or not thumb_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT
            ):
                target_pos = self._previous_pos
            else:
                index_tip_pos = index_tip.pose.position
                index_tip_position = np.array([index_tip_pos.x, index_tip_pos.y, index_tip_pos.z], dtype=np.float32)

                thumb_tip_pos = thumb_tip.pose.position
                thumb_tip_position = np.array([thumb_tip_pos.x, thumb_tip_pos.y, thumb_tip_pos.z], dtype=np.float32)

                target_pos = (index_tip_position + thumb_tip_position) / 2

        self._previous_pos = target_pos

        # rotation
        if self._use_wrist_rotation:
            if not wrist.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT:
                target_rot = self._previous_rot.as_quat()
            else:
                wrist_quat = wrist.pose.orientation
                wrist_quat = np.array([wrist_quat.x, wrist_quat.y, wrist_quat.z, wrist_quat.w], dtype=np.float32)
                target_rot = wrist_quat
                self._previous_rot = Rotation.from_quat(wrist_quat)

        else:
            # use the rotation based on the average position of the index tip and thumb tip
            if (
                not index_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
                or not thumb_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
            ):
                target_rot = self._previous_rot.as_quat()
            else:
                index_tip_quat = index_tip.pose.orientation
                index_tip_quat = np.array(
                    [index_tip_quat.x, index_tip_quat.y, index_tip_quat.z, index_tip_quat.w], dtype=np.float32
                )

                thumb_tip_quat = thumb_tip.pose.orientation
                thumb_tip_quat = np.array(
                    [thumb_tip_quat.x, thumb_tip_quat.y, thumb_tip_quat.z, thumb_tip_quat.w], dtype=np.float32
                )

                r0 = Rotation.from_quat(index_tip_quat)
                r1 = Rotation.from_quat(thumb_tip_quat)
                key_times = [0, 1]
                slerp = Slerp(key_times, Rotation.concatenate([r0, r1]))
                interp_time = [0.5]
                # thumb tip is rotated about the z axis as seen here
                # https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints
                # which requires us to rotate the final pose to align the EE more naturally with the pinch gesture
                interp_rotation = slerp(interp_time)[0] * Rotation.from_euler("x", 90, degrees=True)

                self._previous_rot = interp_rotation
                target_rot = interp_rotation.as_quat()

        if self._zero_out_xy_rotation:
            z, y, x = Rotation.from_quat(target_rot).as_euler("ZYX")
            y = 0.0  # Zero out rotation around y-axis
            x = 0.0  # Zero out rotation around x-axis
            target_rot = (
                Rotation.from_euler("ZYX", [z, y, x]) * Rotation.from_euler("X", np.pi, degrees=False)
            ).as_quat()

        target_rot = [target_rot[3], target_rot[0], target_rot[1], target_rot[2]]

        return np.concatenate([target_pos, target_rot])

    def _calculate_target_delta_pose(self, hand_joints):
        if hand_joints is None:
            return np.zeros(6)

        index_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_INDEX_TIP_EXT]
        thumb_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_THUMB_TIP_EXT]
        wrist = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_WRIST_EXT]

        # Define thresholds
        POSITION_THRESHOLD = 0.001
        ROTATION_THRESHOLD = 0.01

        # position:
        if self._use_wrist_position:
            if not wrist.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT:
                delta_pos = np.zeros(3)
            else:
                wrist_pos = wrist.pose.position
                target_pos = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z], dtype=np.float32)

                delta_pos = target_pos - self._previous_pos

        else:
            if (
                not index_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT
                or not thumb_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT
            ):
                target_pos = self._previous_pos
            else:
                index_tip_pos = index_tip.pose.position
                index_tip_position = np.array([index_tip_pos.x, index_tip_pos.y, index_tip_pos.z], dtype=np.float32)

                thumb_tip_pos = thumb_tip.pose.position
                thumb_tip_position = np.array([thumb_tip_pos.x, thumb_tip_pos.y, thumb_tip_pos.z], dtype=np.float32)

                target_pos = (index_tip_position + thumb_tip_position) / 2
                delta_pos = target_pos - self._previous_pos

        self._previous_pos = target_pos
        self._smoothed_delta_pos = (self._alpha_pos * delta_pos) + ((1 - self._alpha_pos) * self._smoothed_delta_pos)

        # Apply position threshold
        if np.linalg.norm(self._smoothed_delta_pos) < POSITION_THRESHOLD:
            self._smoothed_delta_pos = np.zeros(3)

        # rotation
        if self._use_wrist_rotation:
            # Rotation using wrist orientation only
            if not wrist.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT:
                delta_rot = np.zeros(3)
            else:
                wrist_quat = wrist.pose.orientation
                wrist_quat = np.array([wrist_quat.x, wrist_quat.y, wrist_quat.z, wrist_quat.w], dtype=np.float32)

                wrist_rotation = Rotation.from_quat(wrist_quat)
                target_rot = wrist_rotation.as_rotvec()

                delta_rot = target_rot - self._previous_rot.as_rotvec()
                self._previous_rot = wrist_rotation
        else:
            if (
                not index_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
                or not thumb_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
            ):
                delta_rot = np.zeros(3)
            else:
                index_tip_quat = index_tip.pose.orientation
                index_tip_quat = np.array(
                    [index_tip_quat.x, index_tip_quat.y, index_tip_quat.z, index_tip_quat.w], dtype=np.float32
                )

                thumb_tip_quat = thumb_tip.pose.orientation
                thumb_tip_quat = np.array(
                    [thumb_tip_quat.x, thumb_tip_quat.y, thumb_tip_quat.z, thumb_tip_quat.w], dtype=np.float32
                )

                r0 = Rotation.from_quat(index_tip_quat)
                r1 = Rotation.from_quat(thumb_tip_quat)
                key_times = [0, 1]
                slerp = Slerp(key_times, Rotation.concatenate([r0, r1]))
                interp_time = [0.5]
                interp_rotation = slerp(interp_time)[0]

                relative_rotation = interp_rotation * self._previous_rot.inv()
                delta_rot = relative_rotation.as_rotvec()
                self._previous_rot = interp_rotation

        if self._zero_out_xy_rotation:
            delta_rot[0] = 0.0  # Zero out rotation around x-axis
            delta_rot[1] = 0.0  # Zero out rotation around y-axis

        self._smoothed_delta_rot = (self._alpha_rot * delta_rot) + ((1 - self._alpha_rot) * self._smoothed_delta_rot)

        # Apply rotation threshold
        if np.linalg.norm(self._smoothed_delta_rot) < ROTATION_THRESHOLD:
            self._smoothed_delta_rot = np.zeros(3)

        delta_pos = self._smoothed_delta_pos
        delta_rot = self._smoothed_delta_rot

        # if not tracking still update prev positions but return no delta pose
        if not self._tracking:
            return np.zeros(6)

        return np.concatenate([delta_pos * self._delta_pos_scale_factor, delta_rot * self._delta_rot_scale_factor])

    def _calculate_gripper_command(self, hand_joints):
        if hand_joints is None:
            return self._previous_gripper_command

        index_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_INDEX_TIP_EXT]
        thumb_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_THUMB_TIP_EXT]

        if not self._tracking:
            return self._previous_gripper_command
        if not index_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT:
            return self._previous_gripper_command
        if not thumb_tip.locationFlags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT:
            return self._previous_gripper_command

        index_tip_pos = index_tip.pose.position
        index_tip_pos = np.array([index_tip_pos.x, index_tip_pos.y, index_tip_pos.z], dtype=np.float32)
        thumb_tip_pos = thumb_tip.pose.position
        thumb_tip_pos = np.array([thumb_tip_pos.x, thumb_tip_pos.y, thumb_tip_pos.z], dtype=np.float32)
        distance = np.linalg.norm(index_tip_pos - thumb_tip_pos)
        if distance > self._previous_grip_distance + self.GRIP_HYSTERESIS_METERS:
            self._previous_grip_distance = distance
            self._previous_gripper_command = False
        elif distance < self._previous_grip_distance - self.GRIP_HYSTERESIS_METERS:
            self._previous_grip_distance = distance
            self._previous_gripper_command = True

        return self._previous_gripper_command

    def _on_teleop_command(self, event: carb.events.IEvent):
        msg = event.payload["message"]

        if teleop_command.Actions.START in msg:
            if "START" in self._additional_callbacks:
                self._additional_callbacks["START"]()
            self._tracking = True
        elif teleop_command.Actions.STOP in msg:
            if "STOP" in self._additional_callbacks:
                self._additional_callbacks["STOP"]()
            self._tracking = False
        elif teleop_command.Actions.RESET in msg:
            if "RESET" in self._additional_callbacks:
                self._additional_callbacks["RESET"]()
