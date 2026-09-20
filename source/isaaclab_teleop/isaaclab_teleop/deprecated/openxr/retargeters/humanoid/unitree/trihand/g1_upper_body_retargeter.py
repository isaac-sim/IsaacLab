# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# This import exception is suppressed because g1_dex_retargeting_utils depends
# on pinocchio which is not available on Windows.
with contextlib.suppress(Exception):
    from .g1_dex_retargeting_utils import G1TriHandDexRetargeting


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
        super().__init__(cfg)

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
        left_hand_poses = data[DeviceBase.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[DeviceBase.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        # Handle case where wrist data is not available
        if left_wrist is None or right_wrist is None:
            # Set to default pose if no data available.
            # pos=(0,0,0), quat=(0,0,0,1) (x,y,z,w)
            default_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
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

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

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
            combined_quat = torch.tensor([0, 0.7071, 0, 0.7071], dtype=torch.float32)
        else:
            # Corresponds to a rotation of (0, -90, -90) in euler angles (x,y,z)
            combined_quat = torch.tensor([-0.7071, 0, 0.7071, 0], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])


@dataclass
class G1TriHandUpperBodyRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 Controller Upper Body retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None  # List of robot hand joint names
    retargeter_type: type[RetargeterBase] = G1TriHandUpperBodyRetargeter
