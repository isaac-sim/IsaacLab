# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# This import exception is suppressed because gr1_t2_dex_retargeting_utils depends on pinocchio which is not available on windows
with contextlib.suppress(Exception):
    from .gr1_t2_dex_retargeting_utils import GR1TR2DexRetargeting


@dataclass
class GR1T2RetargeterCfg(RetargeterCfg):
    """Configuration for the GR1T2 retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 100
    hand_joint_names: list[str] | None = None  # List of robot hand joint names


class GR1T2Retargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to GR1T2 hand end-effector commands.

    This retargeter maps hand tracking data from OpenXR to joint commands for the GR1T2 robot's hands.
    It handles both left and right hands, converting poses of the hands in OpenXR format joint angles for the GR1T2 robot's hands.
    """

    def __init__(
        self,
        cfg: GR1T2RetargeterCfg,
    ):
        """Initialize the GR1T2 hand retargeter.

        Args:
            enable_visualization: If True, visualize tracked hand joints
            num_open_xr_hand_joints: Number of joints tracked by OpenXR
            device: PyTorch device for computations
            hand_joint_names: List of robot hand joint names
        """

        self._hand_joint_names = cfg.hand_joint_names
        self._hands_controller = GR1TR2DexRetargeting(self._hand_joint_names)

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
        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]

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
        left_wrist_tensor = torch.tensor(left_wrist, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(self._retarget_abs(right_wrist), dtype=torch.float32, device=self._sim_device)
        hand_joints_tensor = torch.tensor(retargeted_hand_joints, dtype=torch.float32, device=self._sim_device)

        # Combine all tensors into a single tensor
        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def _retarget_abs(self, wrist: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            wrist: Wrist pose data from OpenXR

        Returns:
            Retargeted wrist pose in USD control frame
        """

        # Convert wrist data in openxr frame to usd control frame

        # Create pose object for openxr_right_wrist_in_world
        # Note: The pose utils require torch tensors
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)
        openxr_right_wrist_in_world = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))

        # The usd control frame is 180 degrees rotated around z axis wrt to the openxr frame
        # This was determined through trial and error
        zero_pos = torch.zeros(3, dtype=torch.float32)
        # 180 degree rotation around z axis
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
