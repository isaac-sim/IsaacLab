# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import numpy as np
import torch
import yaml
from dataclasses import MISSING, dataclass
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path

# Helper to import dex_retargeting which might not be available on all platforms
with contextlib.suppress(ImportError):
    from dex_retargeting.retargeting_config import RetargetingConfig


class DexHandRetargeter(RetargeterBase):
    """Generic retargeter for dexterous hands using the dex_retargeting library.

    This class wraps the dex_retargeting library to retarget OpenXR hand tracking data
    to robot hand joint angles. It supports configuration via YAML files and URDFs.
    """

    def __init__(self, cfg: DexHandRetargeterCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """
        super().__init__(cfg)

        # Check if dex_retargeting is available
        if "dex_retargeting" not in globals() and "RetargetingConfig" not in globals():
            raise ImportError("The 'dex_retargeting' package is required but not installed.")

        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names
        self._target = cfg.target

        # Setup paths
        self._prepare_configs(cfg)

        # Initialize dex retargeting optimizer
        self._dex_hand = RetargetingConfig.load_from_file(cfg.hand_retargeting_config).build()

        # Cache joint names from optimizer
        self._dof_names = self._dex_hand.optimizer.robot.dof_joint_names

        # Store transforms
        self._handtracking2baselink = np.array(cfg.handtracking_to_baselink_frame_transform).reshape(3, 3)

        # Map OpenXR joints (26) to Dex-retargeting (21)
        # Indices: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
        # OpenXR: Wrist(1), Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)
        # Note: We skip the metacarpal for non-thumb fingers in OpenXR if needed,
        # but here we use the standard mapping from existing utils.
        self._hand_joints_index = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

        # Visualization
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path=f"/Visuals/dex_hand_markers_{self._target.name}",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        """Process input data and return retargeted joint angles.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: Tensor of hand joint angles.
        """
        hand_poses = data.get(self._target)

        # Visualization
        if self._enable_visualization and hand_poses:
            # Visualize the raw tracked points
            joints_position = np.array([pose[:3] for pose in hand_poses.values()])
            self._markers.visualize(
                translations=torch.tensor(joints_position, device=self._sim_device, dtype=torch.float32)
            )

        # Compute
        q = self._compute_hand(hand_poses, self._dex_hand, self._handtracking2baselink)

        # Map to output vector based on configured joint names
        retargeted_joints = np.zeros(len(self._hand_joint_names))

        for i, name in enumerate(self._dof_names):
            if name in self._hand_joint_names:
                idx = self._hand_joint_names.index(name)
                retargeted_joints[idx] = q[i]

        return torch.tensor(retargeted_joints, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    # -- Internal Helpers --

    def _prepare_configs(self, cfg: DexHandRetargeterCfg):
        """Downloads URDFs if needed and updates YAML config files."""
        # Retrieve URDF (downloads if URL, returns path if local)
        local_urdf = retrieve_file_path(cfg.hand_urdf, force_download=True)

        # Update YAML
        self._update_yaml(cfg.hand_retargeting_config, local_urdf)

    def _update_yaml(self, yaml_path: str, urdf_path: str):
        """Updates the 'urdf_path' field in the retargeting YAML config."""
        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            if "retargeting" in config:
                config["retargeting"]["urdf_path"] = urdf_path
                with open(yaml_path, "w") as f:
                    yaml.dump(config, f)
        except Exception as e:
            print(f"[DexHandRetargeter] Error updating YAML {yaml_path}: {e}")

    def _compute_hand(self, poses: dict | None, retargeting: RetargetingConfig, op2mano: np.ndarray) -> np.ndarray:
        """Computes retargeting for a single hand."""
        if poses is None:
            return np.zeros(len(retargeting.optimizer.robot.dof_joint_names))

        # 1. Extract positions for relevant joints
        hand_joints = list(poses.values())
        joint_pos = np.zeros((21, 3))
        for i, idx in enumerate(self._hand_joints_index):
            joint_pos[i] = hand_joints[idx][:3]

        # 2. Transform to canonical frame
        # Center at wrist (index 0 of our subset, which maps to OpenXR 'wrist' at index 1)
        joint_pos = joint_pos - joint_pos[0:1, :]

        # Apply wrist rotation alignment (OpenXR w,x,y,z -> Scipy x,y,z,w)
        wrist_pose = poses.get("wrist")
        if wrist_pose is None:
            return np.zeros(len(retargeting.optimizer.robot.dof_joint_names))
        xr_wrist_quat = wrist_pose[3:]
        wrist_rot = R.from_quat([xr_wrist_quat[1], xr_wrist_quat[2], xr_wrist_quat[3], xr_wrist_quat[0]]).as_matrix()

        target_pos = joint_pos @ wrist_rot @ op2mano

        # 3. Compute reference value for optimizer
        indices = retargeting.optimizer.target_link_human_indices
        if retargeting.optimizer.retargeting_type == "POSITION":
            ref_value = target_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = target_pos[task_indices, :] - target_pos[origin_indices, :]

        # 4. Run optimizer
        with torch.enable_grad(), torch.inference_mode(False):
            return retargeting.retarget(ref_value)


@dataclass(kw_only=True)
class DexHandRetargeterCfg(RetargeterCfg):
    """Configuration for the generic dexterous hand retargeter."""

    retargeter_type: type[RetargeterBase] = DexHandRetargeter

    # Target Hand
    target: DeviceBase.TrackingTarget = MISSING

    # Joint Names
    hand_joint_names: list[str] = MISSING

    # Config Paths
    hand_retargeting_config: str = MISSING
    hand_urdf: str = MISSING

    # Transforms (3x3 matrix flattened to 9 elements)
    # Default: G1/Inspire coordinate frame
    handtracking_to_baselink_frame_transform: tuple[float, ...] = (0, 0, 1, 1, 0, 0, 0, 1, 0)

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 26


class DexBiManualRetargeter(RetargeterBase):
    """Wrapper around two DexHandRetargeters to support bimanual retargeting with custom joint ordering.

    This class instantiates two :class:`DexHandRetargeter` instances (one for each hand) and combines
    their outputs into a single vector ordered according to `target_joint_names`.
    """

    def __init__(self, cfg: DexBiManualRetargeterCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """
        super().__init__(cfg)
        self._sim_device = cfg.sim_device

        # Initialize individual retargeters
        self._left_retargeter = DexHandRetargeter(cfg.left_hand_cfg)
        self._right_retargeter = DexHandRetargeter(cfg.right_hand_cfg)

        self._target_joint_names = cfg.target_joint_names

        # Prepare index mapping for fast runtime reordering
        self._left_indices = []
        self._right_indices = []
        self._output_indices_left = []
        self._output_indices_right = []

        left_joints = cfg.left_hand_cfg.hand_joint_names
        right_joints = cfg.right_hand_cfg.hand_joint_names

        for i, name in enumerate(self._target_joint_names):
            if name in left_joints:
                self._output_indices_left.append(i)
                self._left_indices.append(left_joints.index(name))
            elif name in right_joints:
                self._output_indices_right.append(i)
                self._right_indices.append(right_joints.index(name))
            else:
                pass

        # Convert to tensors for advanced indexing
        self._left_src_idx = torch.tensor(self._left_indices, device=self._sim_device, dtype=torch.long)
        self._left_dst_idx = torch.tensor(self._output_indices_left, device=self._sim_device, dtype=torch.long)

        self._right_src_idx = torch.tensor(self._right_indices, device=self._sim_device, dtype=torch.long)
        self._right_dst_idx = torch.tensor(self._output_indices_right, device=self._sim_device, dtype=torch.long)

    def retarget(self, data: dict) -> torch.Tensor:
        """Process input data and return retargeted joint angles.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            torch.Tensor: Tensor of combined hand joint angles.
        """
        left_out = self._left_retargeter.retarget(data)
        right_out = self._right_retargeter.retarget(data)

        # Create output buffer
        combined = torch.zeros(len(self._target_joint_names), device=self._sim_device, dtype=torch.float32)

        if len(self._left_dst_idx) > 0:
            combined[self._left_dst_idx] = left_out[self._left_src_idx]

        if len(self._right_dst_idx) > 0:
            combined[self._right_dst_idx] = right_out[self._right_src_idx]

        return combined

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        reqs = set()
        reqs.update(self._left_retargeter.get_requirements())
        reqs.update(self._right_retargeter.get_requirements())
        return list(reqs)


@dataclass(kw_only=True)
class DexBiManualRetargeterCfg(RetargeterCfg):
    """Configuration for the bimanual dexterous hand retargeter."""

    retargeter_type: type[RetargeterBase] = DexBiManualRetargeter

    # Configurations for individual hands
    left_hand_cfg: DexHandRetargeterCfg = MISSING
    right_hand_cfg: DexHandRetargeterCfg = MISSING

    # Combined joint names for the output
    target_joint_names: list[str] = MISSING
