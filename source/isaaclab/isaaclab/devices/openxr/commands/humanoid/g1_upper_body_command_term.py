# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator for GR1T2 upper body teleoperation."""

from __future__ import annotations

import contextlib
import numpy as np
import torch
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices.hand_tracking_device import Hand, HandTrackingDevice
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

# This import exception is suppressed because g1_dex_retargeting_utils depends on pinocchio which is not available on windows
with contextlib.suppress(Exception):
    from isaaclab.devices.openxr.commands.humanoid.g1_dex_retargeting_utils import G1DexRetargeting

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class G1UpperBodyCommandTermCfg(CommandTermCfg):
    """Configuration for the G1 upper body command term."""

    class_type: type[G1UpperBodyCommandTerm] = field(default_factory=lambda: G1UpperBodyCommandTerm)
    """The associated command term class."""

    device_name: str = MISSING
    """Name of the device attribute on the environment.

    The command term will look for an attribute with this name on the environment instance
    (e.g., `env.teleop_device`) to access the hand tracking device.
    """

    enable_visualization: bool = False
    """Whether to enable visualization of hand joints."""

    num_hand_joints: int = 100
    """Number of joints tracked by the hand tracking device."""

    hand_joint_names: list[str] | None = None
    """List of robot hand joint names."""


class G1UpperBodyCommandTerm(CommandTerm):
    """Command term for G1 upper body teleoperation using a hand tracking device.

    This command term generates upper body control commands for the G1 robot based on
    hand tracking data. It processes hand poses and generates wrist poses and
    joint angles for both hands.
    """

    cfg: G1UpperBodyCommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: G1UpperBodyCommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # Store device name for runtime retrieval
        self._device_name = cfg.device_name

        # Initialize the hands controller
        if cfg.hand_joint_names is not None:
            self._hands_controller = G1DexRetargeting(cfg.hand_joint_names)
            self._hand_joint_names = cfg.hand_joint_names
        else:
            raise ValueError("hand_joint_names must be provided in configuration")

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        self._num_hand_joints = cfg.num_hand_joints
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

        # Buffer to store the calculated command tensor
        self._command_tensor = torch.zeros(self.num_envs, 7 + 7 + len(cfg.hand_joint_names), device=self.device)

        # Add metrics
        self.metrics["hand_tracking_active"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "G1UpperBodyCommandTerm:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tVisualization enabled: {self._enable_visualization}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The upper body command tensor containing wrist poses and hand joint angles.

        Shape is (num_envs, 7 + 7 + num_hand_joints) where:
        - First 7 elements: left wrist pose (position + quaternion)
        - Next 7 elements: right wrist pose (position + quaternion)
        - Remaining elements: hand joint angles for both hands
        """
        return self._command_tensor

    """
    Implementation specific functions.
    """

    def _get_hand_tracking_device(self) -> HandTrackingDevice:
        """Get the hand tracking device from the environment."""
        if not hasattr(self._env, self._device_name):
            raise ValueError(f"Hand tracking device '{self._device_name}' not found in environment")

        device = getattr(self._env, self._device_name)
        if not isinstance(device, HandTrackingDevice):
            raise ValueError(f"Device '{self._device_name}' does not implement HandTrackingDevice interface")

        return device

    def _update_metrics(self):
        """Update the metrics based on the current state."""
        # Check if hand tracking data is available using the interface
        try:
            hand_tracking_device = self._get_hand_tracking_device()
        except (ValueError, AttributeError):
            # No hand tracking device available, ignore for now
            return

        try:
            left_hand_data = hand_tracking_device.get_hand_poses(Hand.LEFT)
            right_hand_data = hand_tracking_device.get_hand_poses(Hand.RIGHT)

            # Check if wrist data is available in both hands
            has_data = (
                "wrist" in left_hand_data
                and left_hand_data["wrist"] is not None
                and "wrist" in right_hand_data
                and right_hand_data["wrist"] is not None
            )
            self.metrics["hand_tracking_active"][:] = float(has_data)
        except (ValueError, KeyError):
            self.metrics["hand_tracking_active"][:] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        # Try to get hand tracking data using the interface, return nominal pose if not available
        try:
            hand_tracking_device = self._get_hand_tracking_device()
        except (ValueError, AttributeError):
            # No hand tracking device available, ignore for now
            return

        # Get full joint data for both hands and convert to numpy arrays for retargeting
        left_hand_poses_lists = hand_tracking_device.get_hand_poses(Hand.LEFT)
        right_hand_poses_lists = hand_tracking_device.get_hand_poses(Hand.RIGHT)

        # Convert lists to numpy arrays for compatibility with retargeting functions
        left_hand_poses = {name: np.array(pose) for name, pose in left_hand_poses_lists.items()}
        right_hand_poses = {name: np.array(pose) for name, pose in right_hand_poses_lists.items()}

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        # Handle case where wrist data is not available
        if left_wrist is None or right_wrist is None:
            # Set to default pose if no data available
            default_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            if left_wrist is None:
                left_wrist = default_pose
            if right_wrist is None:
                right_wrist = default_pose

        # Visualization if enabled
        if self._enable_visualization:
            joints_position = np.zeros((self._num_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self.device))

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
            self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self.device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self.device
        )
        hand_joints_tensor = torch.tensor(retargeted_hand_joints, dtype=torch.float32, device=self.device)

        # Combine all tensors into the command tensor for specified environments
        combined_tensor = torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])
        self._command_tensor[env_ids] = combined_tensor.unsqueeze(0).expand(len(env_ids), -1)

    def _update_command(self):
        """Update the command based on the current state.

        For this command term, the command is updated during resampling and
        doesn't need additional updates between resampling events.
        """
        pass

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Handle absolute pose retargeting for left hand."""
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        if is_left:
            combined_quat = torch.tensor([0.7071, 0, 0.7071, 0], dtype=torch.float32)
        else:
            combined_quat = torch.tensor([0, -0.7071, 0, 0.7071], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])
