# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


@dataclass
class Se3RelRetargeterCfg(RetargeterCfg):
    """Configuration for relative position retargeter."""

    zero_out_xy_rotation: bool = True
    use_wrist_rotation: bool = False
    use_wrist_position: bool = True
    delta_pos_scale_factor: float = 10.0
    delta_rot_scale_factor: float = 10.0
    alpha_pos: float = 0.5
    alpha_rot: float = 0.5
    enable_visualization: bool = False
    bound_hand: OpenXRDevice.TrackingTarget = OpenXRDevice.TrackingTarget.HAND_RIGHT


class Se3RelRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to end-effector commands using relative positioning.

    This retargeter calculates delta poses between consecutive hand joint poses to generate incremental robot movements.
    It can either:
    - Use the wrist position and orientation
    - Use the midpoint between thumb and index finger (pinch position)

    Features:
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Motion smoothing with adjustable parameters
    - Optional visualization of the target end-effector pose
    """

    def __init__(
        self,
        cfg: Se3RelRetargeterCfg,
    ):
        """Initialize the relative motion retargeter.

        Args:
            bound_hand: The hand to track (OpenXRDevice.TrackingTarget.HAND_LEFT or OpenXRDevice.TrackingTarget.HAND_RIGHT)
            zero_out_xy_rotation: If True, ignore rotations around x and y axes, allowing only z-axis rotation
            use_wrist_rotation: If True, use wrist rotation for control instead of averaging finger orientations
            use_wrist_position: If True, use wrist position instead of pinch position (midpoint between fingers)
            delta_pos_scale_factor: Amplification factor for position changes (higher = larger robot movements)
            delta_rot_scale_factor: Amplification factor for rotation changes (higher = larger robot rotations)
            alpha_pos: Position smoothing parameter (0-1); higher values track more closely to input, lower values smooth more
            alpha_rot: Rotation smoothing parameter (0-1); higher values track more closely to input, lower values smooth more
            enable_visualization: If True, show a visual marker representing the target end-effector pose
            device: The device to place the returned tensor on ('cpu' or 'cuda')
        """
        # Store the hand to track
        if cfg.bound_hand not in [OpenXRDevice.TrackingTarget.HAND_LEFT, OpenXRDevice.TrackingTarget.HAND_RIGHT]:
            raise ValueError(
                "bound_hand must be either OpenXRDevice.TrackingTarget.HAND_LEFT or"
                " OpenXRDevice.TrackingTarget.HAND_RIGHT"
            )
        super().__init__(cfg)
        self.bound_hand = cfg.bound_hand

        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._use_wrist_rotation = cfg.use_wrist_rotation
        self._use_wrist_position = cfg.use_wrist_position
        self._delta_pos_scale_factor = cfg.delta_pos_scale_factor
        self._delta_rot_scale_factor = cfg.delta_rot_scale_factor
        self._alpha_pos = cfg.alpha_pos
        self._alpha_rot = cfg.alpha_rot

        # Initialize smoothing state
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)

        # Define thresholds for small movements
        self._position_threshold = 0.001
        self._rotation_threshold = 0.01

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            self._goal_marker.set_visibility(True)
            self._visualization_pos = np.zeros(3)
            self._visualization_rot = np.array([1.0, 0.0, 0.0, 0.0])

        self._previous_thumb_tip = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._previous_index_tip = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._previous_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to robot end-effector command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.
                The joint names are defined in isaaclab.devices.openxr.common.HAND_JOINT_NAMES

        Returns:
            torch.Tensor: 6D tensor containing position (xyz) and rotation vector (rx,ry,rz)
                for the robot end-effector
        """
        # Extract key joint poses from the bound hand
        hand_data = data[self.bound_hand]
        thumb_tip = hand_data.get("thumb_tip")
        index_tip = hand_data.get("index_tip")
        wrist = hand_data.get("wrist")

        delta_thumb_tip = self._calculate_delta_pose(thumb_tip, self._previous_thumb_tip)
        delta_index_tip = self._calculate_delta_pose(index_tip, self._previous_index_tip)
        delta_wrist = self._calculate_delta_pose(wrist, self._previous_wrist)
        ee_command_np = self._retarget_rel(delta_thumb_tip, delta_index_tip, delta_wrist)

        self._previous_thumb_tip = thumb_tip.copy()
        self._previous_index_tip = index_tip.copy()
        self._previous_wrist = wrist.copy()

        # Convert to torch tensor
        ee_command = torch.tensor(ee_command_np, dtype=torch.float32, device=self._sim_device)

        return ee_command

    def _calculate_delta_pose(self, joint_pose: np.ndarray, previous_joint_pose: np.ndarray) -> np.ndarray:
        """Calculate delta pose from previous joint pose.

        Args:
            joint_pose: Current joint pose (position and orientation)
            previous_joint_pose: Previous joint pose for the same joint

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta as axis-angle (rx,ry,rz)
        """
        delta_pos = joint_pose[:3] - previous_joint_pose[:3]
        abs_rotation = Rotation.from_quat([*joint_pose[4:7], joint_pose[3]])
        previous_rot = Rotation.from_quat([*previous_joint_pose[4:7], previous_joint_pose[3]])
        relative_rotation = abs_rotation * previous_rot.inv()
        return np.concatenate([delta_pos, relative_rotation.as_rotvec()])

    def _retarget_rel(self, thumb_tip: np.ndarray, index_tip: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """Handle relative (delta) pose retargeting.

        Args:
            thumb_tip: Delta pose of thumb tip
            index_tip: Delta pose of index tip
            wrist: Delta pose of wrist

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta (rx,ry,rz)
        """
        # Get position
        if self._use_wrist_position:
            position = wrist[:3]
        else:
            position = (thumb_tip[:3] + index_tip[:3]) / 2

        # Get rotation
        if self._use_wrist_rotation:
            rotation = wrist[3:6]  # rx, ry, rz
        else:
            rotation = (thumb_tip[3:6] + index_tip[3:6]) / 2

        # Apply zero_out_xy_rotation regardless of rotation source
        if self._zero_out_xy_rotation:
            rotation[0] = 0  # x-axis
            rotation[1] = 0  # y-axis

        # Smooth and scale position
        self._smoothed_delta_pos = self._alpha_pos * position + (1 - self._alpha_pos) * self._smoothed_delta_pos
        if np.linalg.norm(self._smoothed_delta_pos) < self._position_threshold:
            self._smoothed_delta_pos = np.zeros(3)
        position = self._smoothed_delta_pos * self._delta_pos_scale_factor

        # Smooth and scale rotation
        self._smoothed_delta_rot = self._alpha_rot * rotation + (1 - self._alpha_rot) * self._smoothed_delta_rot
        if np.linalg.norm(self._smoothed_delta_rot) < self._rotation_threshold:
            self._smoothed_delta_rot = np.zeros(3)
        rotation = self._smoothed_delta_rot * self._delta_rot_scale_factor

        # Update visualization if enabled
        if self._enable_visualization:
            # Convert rotation vector to quaternion and combine with current rotation
            delta_quat = Rotation.from_rotvec(rotation).as_quat()  # x, y, z, w format
            current_rot = Rotation.from_quat([self._visualization_rot[1:], self._visualization_rot[0]])
            new_rot = Rotation.from_quat(delta_quat) * current_rot
            self._visualization_pos = self._visualization_pos + position
            # Convert back to w, x, y, z format
            self._visualization_rot = np.array([new_rot.as_quat()[3], *new_rot.as_quat()[:3]])
            self._update_visualization()

        return np.concatenate([position, rotation])

    def _update_visualization(self):
        """Update visualization markers with current pose."""
        if self._enable_visualization:
            trans = np.array([self._visualization_pos])
            quat = Rotation.from_matrix(self._visualization_rot).as_quat()
            rot = np.array([np.array([quat[3], quat[0], quat[1], quat[2]])])
            self._goal_marker.visualize(translations=trans, orientations=rot)
