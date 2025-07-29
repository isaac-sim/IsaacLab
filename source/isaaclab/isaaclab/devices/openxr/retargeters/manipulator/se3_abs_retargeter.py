# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


@dataclass
class Se3AbsRetargeterCfg(RetargeterCfg):
    """Configuration for absolute position retargeter."""

    zero_out_xy_rotation: bool = True
    use_wrist_rotation: bool = False
    use_wrist_position: bool = True
    enable_visualization: bool = False
    bound_hand: OpenXRDevice.TrackingTarget = OpenXRDevice.TrackingTarget.HAND_RIGHT


class Se3AbsRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to end-effector commands using absolute positioning.

    This retargeter maps hand joint poses directly to robot end-effector positions and orientations,
    rather than using relative movements. It can either:
    - Use the wrist position and orientation
    - Use the midpoint between thumb and index finger (pinch position)

    Features:
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Optional visualization of the target end-effector pose
    """

    def __init__(
        self,
        cfg: Se3AbsRetargeterCfg,
    ):
        """Initialize the retargeter.

        Args:
            bound_hand: The hand to track (OpenXRDevice.TrackingTarget.HAND_LEFT or OpenXRDevice.TrackingTarget.HAND_RIGHT)
            zero_out_xy_rotation: If True, zero out rotation around x and y axes
            use_wrist_rotation: If True, use wrist rotation instead of finger average
            use_wrist_position: If True, use wrist position instead of pinch position
            enable_visualization: If True, visualize the target pose in the scene
            device: The device to place the returned tensor on ('cpu' or 'cuda')
        """
        super().__init__(cfg)
        if cfg.bound_hand not in [OpenXRDevice.TrackingTarget.HAND_LEFT, OpenXRDevice.TrackingTarget.HAND_RIGHT]:
            raise ValueError(
                "bound_hand must be either OpenXRDevice.TrackingTarget.HAND_LEFT or"
                " OpenXRDevice.TrackingTarget.HAND_RIGHT"
            )
        self.bound_hand = cfg.bound_hand

        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._use_wrist_rotation = cfg.use_wrist_rotation
        self._use_wrist_position = cfg.use_wrist_position

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            self._goal_marker.set_visibility(True)
            self._visualization_pos = np.zeros(3)
            self._visualization_rot = np.array([1.0, 0.0, 0.0, 0.0])

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to robot end-effector command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.
                The joint names are defined in isaaclab.devices.openxr.common.HAND_JOINT_NAMES

        Returns:
            torch.Tensor: 7D tensor containing position (xyz) and orientation (quaternion)
                for the robot end-effector
        """
        # Extract key joint poses from the bound hand
        hand_data = data[self.bound_hand]
        thumb_tip = hand_data.get("thumb_tip")
        index_tip = hand_data.get("index_tip")
        wrist = hand_data.get("wrist")

        ee_command_np = self._retarget_abs(thumb_tip, index_tip, wrist)

        # Convert to torch tensor
        ee_command = torch.tensor(ee_command_np, dtype=torch.float32, device=self._sim_device)

        return ee_command

    def _retarget_abs(self, thumb_tip: np.ndarray, index_tip: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            thumb_tip: 7D array containing position (xyz) and orientation (quaternion)
                for the thumb tip
            index_tip: 7D array containing position (xyz) and orientation (quaternion)
                for the index tip
            wrist: 7D array containing position (xyz) and orientation (quaternion)
                for the wrist

        Returns:
            np.ndarray: 7D array containing position (xyz) and orientation (quaternion)
                for the robot end-effector
        """

        # Get position
        if self._use_wrist_position:
            position = wrist[:3]
        else:
            position = (thumb_tip[:3] + index_tip[:3]) / 2

        # Get rotation
        if self._use_wrist_rotation:
            # wrist is w,x,y,z but scipy expects x,y,z,w
            base_rot = Rotation.from_quat([*wrist[4:], wrist[3]])
        else:
            # Average the orientations of thumb and index using SLERP
            # thumb_tip is w,x,y,z but scipy expects x,y,z,w
            r0 = Rotation.from_quat([*thumb_tip[4:], thumb_tip[3]])
            # index_tip is w,x,y,z but scipy expects x,y,z,w
            r1 = Rotation.from_quat([*index_tip[4:], index_tip[3]])
            key_times = [0, 1]
            slerp = Slerp(key_times, Rotation.concatenate([r0, r1]))
            base_rot = slerp([0.5])[0]

        # Apply additional x-axis rotation to align with pinch gesture
        final_rot = base_rot * Rotation.from_euler("x", 90, degrees=True)

        if self._zero_out_xy_rotation:
            z, y, x = final_rot.as_euler("ZYX")
            y = 0.0  # Zero out rotation around y-axis
            x = 0.0  # Zero out rotation around x-axis
            final_rot = Rotation.from_euler("ZYX", [z, y, x]) * Rotation.from_euler("X", np.pi, degrees=False)

        # Convert back to w,x,y,z format
        quat = final_rot.as_quat()
        rotation = np.array([quat[3], quat[0], quat[1], quat[2]])  # Output remains w,x,y,z

        # Update visualization if enabled
        if self._enable_visualization:
            self._visualization_pos = position
            self._visualization_rot = rotation
            self._update_visualization()

        return np.concatenate([position, rotation])

    def _update_visualization(self):
        """Update visualization markers with current pose.

        If visualization is enabled, the target end-effector pose is visualized in the scene.
        """
        if self._enable_visualization:
            trans = np.array([self._visualization_pos])
            quat = Rotation.from_matrix(self._visualization_rot).as_quat()
            rot = np.array([np.array([quat[3], quat[0], quat[1], quat[2]])])
            self._goal_marker.visualize(translations=trans, orientations=rot)
