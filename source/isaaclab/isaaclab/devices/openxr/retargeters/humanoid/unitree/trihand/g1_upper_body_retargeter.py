# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.openxr.openxr_device_controller import MotionControllerTrackingTarget, MotionControllerDataRowIndex, MotionControllerInputIndex
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# This import exception is suppressed because g1_dex_retargeting_utils depends on pinocchio which is not available on windows
with contextlib.suppress(Exception):
    from .g1_dex_retargeting_utils import G1TriHandDexRetargeting


@dataclass
class G1TriHandUpperBodyRetargeterCfg(RetargeterCfg):
    """Configuration for the G1UpperBody retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None  # List of robot hand joint names


class G1TriHandUpperBodyRetargeter(RetargeterBase):
    """Retargets OpenXR data to G1 upper body commands.

    This retargeter maps hand tracking data from OpenXR to wrist and hand joint commands for the G1 robot.
    It handles both left and right hands, converting poses of the hands in OpenXR format to appropriate wrist poses
    and joint angles for the G1 robot's upper body.
    """

    def __init__(
        self,
        cfg: G1TriHandUpperBodyRetargeterCfg,
    ):
        """Initialize the G1 upper body retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """

        # Store device name for runtime retrieval
        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names

        # Initialize the hands controller
        if cfg.hand_joint_names is not None:
            self._hands_controller = G1TriHandDexRetargeting(cfg.hand_joint_names)
        else:
            raise ValueError("hand_joint_names must be provided in configuration")

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/g1_hand_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to robot end-effector commands.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            A tensor containing the retargeted commands:
                - Left wrist pose (7)
                - Right wrist pose (7)
                - Hand joint angles (len(hand_joint_names))
        """

        # Access the left and right hand data using the enum key
        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        # Handle case where wrist data is not available
        if left_wrist is None or right_wrist is None:
            # Set to default pose if no data available.
            # pos=(0,0,0), quat=(1,0,0,0) (w,x,y,z)
            default_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            if left_wrist is None:
                left_wrist = default_pose
            if right_wrist is None:
                right_wrist = default_pose

        # Visualization if enabled
        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        # Compute retargeted hand joints
        left_hands_pos = self._hands_controller.compute_left(left_hand_poses)
        indexes = [self._hand_joint_names.index(name) for name in self._hands_controller.get_left_joint_names()]
        left_retargeted_hand_joints = np.zeros(len(self._hands_controller.get_joint_names()))
        left_retargeted_hand_joints[indexes] = left_hands_pos
        left_hand_joints = left_retargeted_hand_joints

        right_hands_pos = self._hands_controller.compute_right(right_hand_poses)
        indexes = [self._hand_joint_names.index(name) for name in self._hands_controller.get_right_joint_names()]
        right_retargeted_hand_joints = np.zeros(len(self._hands_controller.get_joint_names()))
        right_retargeted_hand_joints[indexes] = right_hands_pos
        right_hand_joints = right_retargeted_hand_joints
        retargeted_hand_joints = left_hand_joints + right_hand_joints

        # Convert numpy arrays to tensors and store in command buffer
        left_wrist_tensor = torch.tensor(
            self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self._sim_device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self._sim_device
        )
        hand_joints_tensor = torch.tensor(retargeted_hand_joints, dtype=torch.float32, device=self._sim_device)

        # Combine all tensors into a single tensor
        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            wrist: Wrist pose data from OpenXR.
            is_left: True for the left hand, False for the right hand.

        Returns:
            Retargeted wrist pose in USD control frame.
        """
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        if is_left:
            # Corresponds to a rotation of (0, 90, 90) in euler angles (x,y,z)
            combined_quat = torch.tensor([0.7071, 0, 0.7071, 0], dtype=torch.float32)
        else:
            # Corresponds to a rotation of (0, -90, -90) in euler angles (x,y,z)
            combined_quat = torch.tensor([0, -0.7071, 0, 0.7071], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])


@dataclass
class G1TriHandControllerUpperBodyRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 Controller Upper Body retargeter."""

    enable_visualization: bool = False
    hand_joint_names: list[str] | None = None  # List of robot hand joint names


class G1TriHandControllerUpperBodyRetargeter(RetargeterBase):
    """Simple retargeter that maps motion controller inputs to G1 hand joints.

    Mapping:
    - A button (digital 0/1) → Thumb joints
    - Trigger (analog 0-1) → Index finger joints  
    - Squeeze (analog 0-1) → Middle finger joints
    """

    def __init__(self, cfg: G1TriHandControllerUpperBodyRetargeterCfg):
        """Initialize the retargeter."""
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
            hand_joints order: [left_proximal(3), right_proximal(3), left_distal(2), left_thumb_middle(1), right_distal(2), right_thumb_middle(1), left_thumb_tip(1), right_thumb_tip(1)]
        """


        # Get controller data
        left_controller_data = data.get(MotionControllerTrackingTarget.LEFT, np.array([]))
        right_controller_data = data.get(MotionControllerTrackingTarget.RIGHT, np.array([]))

        # Default wrist poses (position + quaternion)
        default_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # Extract poses from controller data
        left_wrist = self._extract_wrist_pose(left_controller_data, default_wrist)
        right_wrist = self._extract_wrist_pose(right_controller_data, default_wrist)
        
        # Map controller inputs to hand joints
        left_hand_joints = self._map_to_hand_joints(left_controller_data, is_left=True)
        right_hand_joints = self._map_to_hand_joints(right_controller_data, is_left=False)
        
        # Negate left hand joints for proper mirroring
        left_hand_joints = -left_hand_joints
        
        # Combine joints in the expected order: [left_proximal(3), right_proximal(3), left_distal(2), left_thumb_middle(1), right_distal(2), right_thumb_middle(1), left_thumb_tip(1), right_thumb_tip(1)]
        all_hand_joints = np.array([
            left_hand_joints[3],   # left_index_proximal
            left_hand_joints[5],   # left_middle_proximal
            left_hand_joints[0],   # left_thumb_base
            right_hand_joints[3],  # right_index_proximal
            right_hand_joints[5],  # right_middle_proximal
            right_hand_joints[0],  # right_thumb_base
            left_hand_joints[4],   # left_index_distal
            left_hand_joints[6],   # left_middle_distal
            left_hand_joints[1],   # left_thumb_middle
            right_hand_joints[4],  # right_index_distal
            right_hand_joints[6],  # right_middle_distal
            right_hand_joints[1],  # right_thumb_middle
            left_hand_joints[2],   # left_thumb_tip
            right_hand_joints[2],  # right_thumb_tip
        ])
        
        # Convert to tensors
        left_wrist_tensor = torch.tensor(
            self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self._sim_device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self._sim_device
        )
        hand_joints_tensor = torch.tensor(all_hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def _extract_wrist_pose(self, controller_data: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
        """Extract wrist pose from controller data.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            default_pose: Default pose to use if no data

        Returns:
            Wrist pose array [x, y, z, w, x, y, z]
        """
        if len(controller_data) > MotionControllerDataRowIndex.POSE.value:
            return controller_data[MotionControllerDataRowIndex.POSE.value]
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
        
        if len(controller_data) <= MotionControllerDataRowIndex.INPUTS.value:
            return hand_joints
            
        # Extract inputs from second row
        inputs = controller_data[MotionControllerDataRowIndex.INPUTS.value]
        
        if len(inputs) <= MotionControllerInputIndex.BUTTON_0.value:
            return hand_joints
            
        # Extract specific inputs using enum
        trigger = inputs[MotionControllerInputIndex.TRIGGER.value]  # 0.0 to 1.0 (analog)
        squeeze = inputs[MotionControllerInputIndex.SQUEEZE.value]  # 0.0 to 1.0 (analog)

        # Grasping logic: If trigger is pressed, we grasp with index and thumb. If squeeze is pressed, we grasp with middle and thumb.
        # If both are pressed, we grasp with index, middle, and thumb.
        # The thumb rotates towards the direction of the pressing finger. If both are pressed, the thumb stays in the middle.

        thumb_button = max(trigger, squeeze)
        
        # Map to G1 hand joints (in radians)
        # Thumb joints (3 joints) - controlled by A button (digital)
        thumb_angle = -thumb_button  # Max 1 radian ≈ 57°
        
        # Thumb rotation: If trigger is pressed, we rotate the thumb toward the index finger. If squeeze is pressed, we rotate the thumb toward the middle finger.
        # If both are pressed, the thumb stays between the index and middle fingers.
        # Trigger pushes toward +0.5, squeeze pushes toward -0.5
        # trigger=1,squeeze=0 → 0.5; trigger=0,squeeze=1 → -0.5; both=1 → 0
        thumb_rotation = 0.5 * trigger - 0.5 * squeeze

        if not is_left:
            thumb_rotation = -thumb_rotation

        # These values were found empirically to get a good gripper pose.

        hand_joints[0] = thumb_rotation     # thumb_0_joint (base)
        hand_joints[1] = thumb_angle * 0.4  # thumb_1_joint (middle)
        hand_joints[2] = thumb_angle * 0.7  # thumb_2_joint (tip)
        
        # Index finger joints (2 joints) - controlled by trigger (analog)
        index_angle = trigger * 1.0     # Max 1.0 radians ≈ 57°
        hand_joints[3] = index_angle    # index_0_joint (proximal)
        hand_joints[4] = index_angle    # index_1_joint (distal)
        
        # Middle finger joints (2 joints) - controlled by squeeze (analog)
        middle_angle = squeeze * 1.0    # Max 1.0 radians ≈ 57°
        hand_joints[5] = middle_angle   # middle_0_joint (proximal)
        hand_joints[6] = middle_angle   # middle_1_joint (distal)

        return hand_joints

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Handle absolute pose retargeting for controller wrists."""
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        # Combined -75° (rather than -90° for wrist comfort) Y rotation + 90° Z rotation
        # This is equivalent to (0, -75, 90) in euler angles
        combined_quat = torch.tensor([0.5358, -0.4619, 0.5358, 0.4619], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])
