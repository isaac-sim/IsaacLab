# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


@dataclass
class H1_2RetargeterCfg(RetargeterCfg):
    """Configuration for the H1_2 retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None  # List of robot hand joint names
    arm_joint_names: list[str] | None = None  # List of robot arm joint names


class H1_2Retargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to H1_2 hand and arm end-effector commands.

    This retargeter maps hand tracking data from OpenXR to joint commands for the H1_2 robot's hands and arms.
    It handles both left and right hands, converting poses to joint angles for the H1_2 robot.
    """

    def __init__(
        self,
        cfg: H1_2RetargeterCfg,
    ):
        """Initialize the H1_2 hand retargeter.

        Args:
            cfg: Configuration object containing retargeter settings
        """
        super().__init__(cfg)
        
        self._hand_joint_names = cfg.hand_joint_names or []
        self._arm_joint_names = cfg.arm_joint_names or []
        
        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device
        
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to robot end-effector commands.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: Concatenated tensor containing:
                - Left wrist pose (7 values: x, y, z, qx, qy, qz, qw)
                - Right wrist pose (7 values: x, y, z, qx, qy, qz, qw)
                - Left hand joint angles (variable length based on hand_joint_names)
                - Right hand joint angles (variable length based on hand_joint_names)
                - Left arm joint angles (variable length based on arm_joint_names)
                - Right arm joint angles (variable length based on arm_joint_names)
        """

        # Access the left and right hand data using the enum key
        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))

            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])

            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        # Process hand joint angles
        left_hand_joints = self._process_hand_joints(left_hand_poses, "left")
        right_hand_joints = self._process_hand_joints(right_hand_poses, "right")
        
        # Process arm joint angles (simplified IK for arm positioning)
        left_arm_joints = self._process_arm_joints(left_wrist, "left")
        right_arm_joints = self._process_arm_joints(right_wrist, "right")

        # Convert numpy arrays to tensors and concatenate them
        left_wrist_tensor = torch.tensor(left_wrist, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(self._retarget_abs(right_wrist), dtype=torch.float32, device=self._sim_device)
        left_hand_tensor = torch.tensor(left_hand_joints, dtype=torch.float32, device=self._sim_device)
        right_hand_tensor = torch.tensor(right_hand_joints, dtype=torch.float32, device=self._sim_device)
        left_arm_tensor = torch.tensor(left_arm_joints, dtype=torch.float32, device=self._sim_device)
        right_arm_tensor = torch.tensor(right_arm_joints, dtype=torch.float32, device=self._sim_device)

        # Combine all tensors into a single tensor
        return torch.cat([
            left_wrist_tensor, 
            right_wrist_tensor, 
            left_hand_tensor, 
            right_hand_tensor,
            left_arm_tensor,
            right_arm_tensor
        ])

    def _process_hand_joints(self, hand_poses: dict, hand_side: str) -> np.ndarray:
        """Process hand joint poses to generate joint angles.
        
        Args:
            hand_poses: Dictionary of hand joint poses from OpenXR
            hand_side: "left" or "right" to identify which hand
            
        Returns:
            np.ndarray: Joint angles for the hand
        """
        # Create array of zeros with length matching number of joint names
        hand_joints = np.zeros(len(self._hand_joint_names))
        
        # Simple mapping from OpenXR hand joints to H1_2 hand joints
        # This is a simplified implementation - you may need to adjust based on actual H1_2 hand structure
        
        # Map finger joints (simplified)
        finger_mappings = {
            "thumb": ["thumb_tip", "thumb_ip", "thumb_mcp"],
            "index": ["index_tip", "index_dip", "index_pip", "index_mcp"],
            "middle": ["middle_tip", "middle_dip", "middle_pip", "middle_mcp"],
            "ring": ["ring_tip", "ring_dip", "ring_pip", "ring_mcp"],
            "pinky": ["pinky_tip", "pinky_dip", "pinky_pip", "pinky_mcp"]
        }
        
        # Calculate joint angles based on finger positions
        for finger_name, joint_names in finger_mappings.items():
            if finger_name in self._hand_joint_names:
                # Simple angle calculation based on finger tip position relative to wrist
                if "tip" in joint_names[0] and joint_names[0] in hand_poses:
                    tip_pos = hand_poses[joint_names[0]][:3]
                    wrist_pos = hand_poses["wrist"][:3]
                    
                    # Calculate angle based on distance from wrist
                    distance = np.linalg.norm(tip_pos - wrist_pos)
                    # Normalize to reasonable joint angle range
                    angle = np.clip(distance * 2.0, 0.0, 1.0)  # Scale factor may need adjustment
                    
                    joint_idx = self._hand_joint_names.index(finger_name)
                    hand_joints[joint_idx] = angle
        
        return hand_joints

    def _process_arm_joints(self, wrist_pose: np.ndarray, arm_side: str) -> np.ndarray:
        """Process wrist pose to generate arm joint angles using simplified IK.
        
        Args:
            wrist_pose: Wrist pose from OpenXR
            arm_side: "left" or "right" to identify which arm
            
        Returns:
            np.ndarray: Joint angles for the arm
        """
        # Create array of zeros with length matching number of arm joint names
        arm_joints = np.zeros(len(self._arm_joint_names))
        
        # Extract wrist position and orientation
        wrist_pos = wrist_pose[:3]
        wrist_quat = wrist_pose[3:]
        
        # Simple IK calculation for arm joints
        # This is a simplified implementation - you may need more sophisticated IK
        
        # Calculate shoulder and elbow angles based on wrist position
        # Assuming a simple 2-link arm model (shoulder to elbow, elbow to wrist)
        
        # Base position (shoulder position - you may need to adjust this)
        base_pos = np.array([0.0, 0.0, 1.0])  # Approximate shoulder position
        
        # Vector from base to wrist
        target_vector = wrist_pos - base_pos
        
        # Simple angle calculations (this is very simplified)
        # You may need to implement proper IK solver here
        
        # Shoulder pitch (rotation around Y axis)
        shoulder_pitch = np.arctan2(target_vector[2], target_vector[0])
        
        # Shoulder roll (rotation around X axis)
        shoulder_roll = np.arctan2(target_vector[1], np.sqrt(target_vector[0]**2 + target_vector[2]**2))
        
        # Elbow angle (simplified)
        elbow_angle = np.pi / 4  # 45 degrees as default
        
        # Map to actual joint names
        if "shoulder_pitch" in self._arm_joint_names:
            idx = self._arm_joint_names.index("shoulder_pitch")
            arm_joints[idx] = shoulder_pitch
            
        if "shoulder_roll" in self._arm_joint_names:
            idx = self._arm_joint_names.index("shoulder_roll")
            arm_joints[idx] = shoulder_roll
            
        if "elbow" in self._arm_joint_names:
            idx = self._arm_joint_names.index("elbow")
            arm_joints[idx] = elbow_angle
        
        return arm_joints

    def _retarget_abs(self, wrist: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            wrist: Wrist pose data from OpenXR

        Returns:
            Retargeted wrist pose in USD control frame
        """
        # Convert wrist data in openxr frame to usd control frame
        # Similar to GR1T2 implementation but may need adjustments for H1_2

        # Create pose object for openxr_right_wrist_in_world
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)
        openxr_right_wrist_in_world = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))

        # The usd control frame transformation for H1_2
        # This may need adjustment based on H1_2's coordinate system
        zero_pos = torch.zeros(3, dtype=torch.float32)
        # Adjust rotation as needed for H1_2
        z_axis_rot_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        usd_right_roll_link_in_openxr_right_wrist = PoseUtils.make_pose(
            zero_pos, PoseUtils.matrix_from_quat(z_axis_rot_quat)
        )

        # Convert wrist pose in openxr frame to usd control frame
        usd_right_roll_link_in_world = PoseUtils.pose_in_A_to_pose_in_B(
            usd_right_roll_link_in_openxr_right_wrist, openxr_right_wrist_in_world
        )

        # extract position and rotation
        usd_right_roll_link_in_world_pos, usd_right_roll_link_in_world_mat = PoseUtils.unmake_pose(
            usd_right_roll_link_in_world
        )
        usd_right_roll_link_in_world_quat = PoseUtils.quat_from_matrix(usd_right_roll_link_in_world_mat)

        return np.concatenate([usd_right_roll_link_in_world_pos, usd_right_roll_link_in_world_quat])
