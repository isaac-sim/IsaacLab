# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class G1TriHandUpperBodyMotionControllerRetargeter(RetargeterBase):
    """Simple retargeter that maps motion controller inputs to G1 hand joints.

    Mapping:
    - A button (digital 0/1) → Thumb joints
    - Trigger (analog 0-1) → Index finger joints
    - Squeeze (analog 0-1) → Middle finger joints
    """

    def __init__(self, cfg: G1TriHandUpperBodyMotionControllerRetargeterCfg):
        """Initialize the retargeter."""
        super().__init__(cfg)
        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names
        self._enable_visualization = cfg.enable_visualization

        if cfg.hand_joint_names is None:
            raise ValueError("hand_joint_names must be provided")

        # Initialize visualization if enabled
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/g1_controller_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert controller inputs to robot commands.

        Args:
            data: Dictionary with MotionControllerTrackingTarget.LEFT/RIGHT keys
                 Each value is a 2D array: [pose(7), inputs(7)]

        Returns:
            Tensor: [left_wrist(7), right_wrist(7), hand_joints(14)]
            hand_joints order:
                [
                    left_proximal(3), right_proximal(3), left_distal(2), left_thumb_middle(1),
                    right_distal(2), right_thumb_middle(1), left_thumb_tip(1), right_thumb_tip(1)
                ]
        """

        # Get controller data
        left_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        # Default wrist poses (position + quaternion xyzw)
        default_wrist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # Extract poses from controller data
        left_wrist = self._extract_wrist_pose(left_controller_data, default_wrist)
        right_wrist = self._extract_wrist_pose(right_controller_data, default_wrist)

        # Map controller inputs to hand joints
        left_hand_joints = self._map_to_hand_joints(left_controller_data, is_left=True)
        right_hand_joints = self._map_to_hand_joints(right_controller_data, is_left=False)

        # Negate left hand joints for proper mirroring
        left_hand_joints = -left_hand_joints

        # Combine joints in the expected order:
        #   [left_proximal(3), right_proximal(3), left_distal(2), left_thumb_middle(1),
        #    right_distal(2), right_thumb_middle(1), left_thumb_tip(1), right_thumb_tip(1)]
        all_hand_joints = np.array(
            [
                left_hand_joints[3],  # left_index_proximal
                left_hand_joints[5],  # left_middle_proximal
                left_hand_joints[0],  # left_thumb_base
                right_hand_joints[3],  # right_index_proximal
                right_hand_joints[5],  # right_middle_proximal
                right_hand_joints[0],  # right_thumb_base
                left_hand_joints[4],  # left_index_distal
                left_hand_joints[6],  # left_middle_distal
                left_hand_joints[1],  # left_thumb_middle
                right_hand_joints[4],  # right_index_distal
                right_hand_joints[6],  # right_middle_distal
                right_hand_joints[1],  # right_thumb_middle
                left_hand_joints[2],  # left_thumb_tip
                right_hand_joints[2],  # right_thumb_tip
            ]
        )

        # Convert to tensors
        left_wrist_tensor = torch.tensor(
            self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self._sim_device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self._sim_device
        )
        hand_joints_tensor = torch.tensor(all_hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _extract_wrist_pose(self, controller_data: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
        """Extract wrist pose from controller data.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            default_pose: Default pose to use if no data

        Returns:
            Wrist pose array [x, y, z, w, x, y, z]
        """
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        return default_pose

    def _map_to_hand_joints(self, controller_data: np.ndarray, is_left: bool) -> np.ndarray:
        """Map controller inputs to hand joint angles.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            is_left: True for left hand, False for right hand

        Returns:
            Hand joint angles (7 joints per hand) in radians
        """

        # Initialize all joints to zero
        hand_joints = np.zeros(7)

        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return hand_joints

        # Extract inputs from second row
        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]

        if len(inputs) < len(DeviceBase.MotionControllerInputIndex):
            return hand_joints

        # Extract specific inputs using enum
        trigger = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]  # 0.0 to 1.0 (analog)
        squeeze = inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]  # 0.0 to 1.0 (analog)

        # Grasping logic:
        #   If trigger is pressed, we grasp with index and thumb.
        #   If squeeze is pressed, we grasp with middle and thumb.
        #   If both are pressed, we grasp with index, middle, and thumb.
        # The thumb rotates towards the direction of the pressing finger.
        #   If both are pressed, the thumb stays in the middle.

        thumb_button = max(trigger, squeeze)

        # Map to G1 hand joints (in radians)
        # Thumb joints (3 joints) - controlled by A button (digital)
        thumb_angle = -thumb_button  # Max 1 radian ≈ 57°

        # Thumb rotation:
        #   If trigger is pressed, we rotate the thumb toward the index finger.
        #   If squeeze is pressed, we rotate the thumb toward the middle finger.
        #   If both are pressed, the thumb stays between the index and middle fingers.
        # Trigger pushes toward +0.5, squeeze pushes toward -0.5
        # trigger=1,squeeze=0 → 0.5; trigger=0,squeeze=1 → -0.5; both=1 → 0
        thumb_rotation = 0.5 * trigger - 0.5 * squeeze

        if not is_left:
            thumb_rotation = -thumb_rotation

        # These values were found empirically to get a good gripper pose.

        hand_joints[0] = thumb_rotation  # thumb_0_joint (base)
        hand_joints[1] = thumb_angle * 0.4  # thumb_1_joint (middle)
        hand_joints[2] = thumb_angle * 0.7  # thumb_2_joint (tip)

        # Index finger joints (2 joints) - controlled by trigger (analog)
        index_angle = trigger * 1.0  # Max 1.0 radians ≈ 57°
        hand_joints[3] = index_angle  # index_0_joint (proximal)
        hand_joints[4] = index_angle  # index_1_joint (distal)

        # Middle finger joints (2 joints) - controlled by squeeze (analog)
        middle_angle = squeeze * 1.0  # Max 1.0 radians ≈ 57°
        hand_joints[5] = middle_angle  # middle_0_joint (proximal)
        hand_joints[6] = middle_angle  # middle_1_joint (distal)

        return hand_joints

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Handle absolute pose retargeting for controller wrists."""
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        # Combined -75° (rather than -90° for wrist comfort) Y rotation + 90° Z rotation
        # This is equivalent to (0, -75, 90) in euler angles
        combined_quat = torch.tensor([-0.4619, 0.5358, 0.4619, 0.5358], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])


@dataclass
class G1TriHandUpperBodyMotionControllerRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 Controller Upper Body retargeter."""

    enable_visualization: bool = False
    hand_joint_names: list[str] | None = None  # List of robot hand joint names
    retargeter_type: type[RetargeterBase] = G1TriHandUpperBodyMotionControllerRetargeter
