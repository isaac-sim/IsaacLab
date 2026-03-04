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

# This import exception is suppressed because g1_dex_retargeting_utils
# depends on pinocchio which is not available on Windows.
with contextlib.suppress(Exception):
    from .g1_dex_retargeting_utils import UnitreeG1DexRetargeting


class UnitreeG1Retargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to GR1T2 hand end-effector commands.

    This retargeter maps hand tracking data from OpenXR to joint commands for the GR1T2 robot's hands.
    It handles both left and right hands, converting poses of the hands in OpenXR format joint angles
    for the GR1T2 robot's hands.
    """

    def __init__(
        self,
        cfg: UnitreeG1RetargeterCfg,
    ):
        """Initialize the UnitreeG1 hand retargeter.

        Args:
            enable_visualization: If True, visualize tracked hand joints
            num_open_xr_hand_joints: Number of joints tracked by OpenXR
            device: PyTorch device for computations
            hand_joint_names: List of robot hand joint names
        """

        super().__init__(cfg)
        self._hand_joint_names = cfg.hand_joint_names
        self._hands_controller = UnitreeG1DexRetargeting(self._hand_joint_names)

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
            tuple containing:
                Left wrist pose
                Right wrist pose in USD frame
                Retargeted hand joint angles
        """

        # Access the left and right hand data using the enum key
        left_hand_poses = data[DeviceBase.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[DeviceBase.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))

            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])

            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        # Create array of zeros with length matching number of joint names
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

        # Convert numpy arrays to tensors and concatenate them
        left_wrist_tensor = torch.tensor(
            self._retarget_abs(left_wrist, True), dtype=torch.float32, device=self._sim_device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, False), dtype=torch.float32, device=self._sim_device
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
        # Note: This was determined through trial, use the target quat and cloudXR quat,
        # to estimate a most reasonable transformation matrix

        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        if is_left:
            # Corresponds to a rotation of (0, 180, 0) in euler angles (x,y,z)
            combined_quat = torch.tensor([0, 0.7071, 0, 0.7071], dtype=torch.float32)
        else:
            # Corresponds to a rotation of (180, 0, 0) in euler angles (x,y,z)
            combined_quat = torch.tensor([0.7071, 0, -0.7071, 0], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])


@dataclass
class UnitreeG1RetargeterCfg(RetargeterCfg):
    """Configuration for the UnitreeG1 retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None  # List of robot hand joint names
    retargeter_type: type[RetargeterBase] = UnitreeG1Retargeter
