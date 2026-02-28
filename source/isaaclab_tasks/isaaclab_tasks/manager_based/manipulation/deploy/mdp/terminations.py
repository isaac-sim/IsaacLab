# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based termination terms specific to the gear assembly manipulation environments."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .events import randomize_gear_type


class reset_when_gear_dropped(ManagerTermBase):
    """Check if the gear has fallen out of the gripper and return reset flags.

    This class-based term pre-caches all required tensors and gear offsets.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        """Initialize the reset when gear dropped term.

        Args:
            cfg: Termination term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get robot asset configuration
        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        # Validate required parameters
        if "end_effector_body_name" not in cfg.params:
            raise ValueError(
                "'end_effector_body_name' parameter is required in reset_when_gear_dropped configuration. "
                "Example: 'wrist_3_link'"
            )
        if "grasp_rot_offset" not in cfg.params:
            raise ValueError(
                "'grasp_rot_offset' parameter is required in reset_when_gear_dropped configuration. "
                "It should be a quaternion [x, y, z, w]. Example: [0.707, 0.707, 0.0, 0.0]"
            )

        self.end_effector_body_name = cfg.params["end_effector_body_name"]

        # Pre-cache gear grasp offsets as tensors (required parameter)
        if "gear_offsets_grasp" not in cfg.params:
            raise ValueError(
                "'gear_offsets_grasp' parameter is required in reset_when_gear_dropped configuration. "
                "It should be a dict with keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )
        gear_offsets_grasp = cfg.params["gear_offsets_grasp"]
        if not isinstance(gear_offsets_grasp, dict):
            raise TypeError(
                f"'gear_offsets_grasp' parameter must be a dict, got {type(gear_offsets_grasp).__name__}. "
                "It should have keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )

        self.gear_grasp_offset_tensors = {}
        for gear_type in ["gear_small", "gear_medium", "gear_large"]:
            if gear_type not in gear_offsets_grasp:
                raise ValueError(
                    f"'{gear_type}' offset is required in 'gear_offsets_grasp' parameter. "
                    f"Found keys: {list(gear_offsets_grasp.keys())}"
                )
            self.gear_grasp_offset_tensors[gear_type] = torch.tensor(
                gear_offsets_grasp[gear_type], device=env.device, dtype=torch.float32
            )

        # Stack grasp offset tensors for vectorized indexing (shape: 3, 3)
        # Index 0=small, 1=medium, 2=large
        self.gear_grasp_offsets_stacked = torch.stack(
            [
                self.gear_grasp_offset_tensors["gear_small"],
                self.gear_grasp_offset_tensors["gear_medium"],
                self.gear_grasp_offset_tensors["gear_large"],
            ],
            dim=0,
        )

        # Pre-cache grasp rotation offset tensor
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        # Pre-allocate buffers
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)
        self.gear_grasp_offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self.all_gear_pos_buffer = torch.zeros(env.num_envs, 3, 3, device=env.device, dtype=torch.float32)
        self.all_gear_quat_buffer = torch.zeros(env.num_envs, 3, 4, device=env.device, dtype=torch.float32)
        self.reset_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

        # Find end effector index once
        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            logger.warning(
                f"{self.end_effector_body_name} not found in robot body names. Cannot check gear drop condition."
            )
            self.eef_idx = None
        else:
            self.eef_idx = eef_indices[0]

    def __call__(
        self,
        env: ManagerBasedEnv,
        distance_threshold: float = 0.1,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gear_offsets_grasp: dict | None = None,
        end_effector_body_name: str | None = None,
        grasp_rot_offset: list | None = None,
    ) -> torch.Tensor:
        """Check if gear has dropped and return reset flags.

        Args:
            env: Environment instance
            distance_threshold: Maximum allowed distance between gear grasp point and gripper
            robot_asset_cfg: Configuration for the robot asset (unused, kept for compatibility)

        Returns:
            Boolean tensor indicating which environments should be reset
        """
        # Reset flags
        self.reset_flags.fill_(False)

        if self.eef_idx is None:
            return self.reset_flags

        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this termination term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Get end effector position
        eef_pos_world = wp.to_torch(self.robot_asset.data.body_link_pos_w)[:, self.eef_idx]

        # Update gear positions and quaternions in buffers
        self.all_gear_pos_buffer[:, 0, :] = wp.to_torch(self.gear_assets["gear_small"].data.root_link_pos_w)
        self.all_gear_pos_buffer[:, 1, :] = wp.to_torch(self.gear_assets["gear_medium"].data.root_link_pos_w)
        self.all_gear_pos_buffer[:, 2, :] = wp.to_torch(self.gear_assets["gear_large"].data.root_link_pos_w)

        self.all_gear_quat_buffer[:, 0, :] = wp.to_torch(self.gear_assets["gear_small"].data.root_link_quat_w)
        self.all_gear_quat_buffer[:, 1, :] = wp.to_torch(self.gear_assets["gear_medium"].data.root_link_quat_w)
        self.all_gear_quat_buffer[:, 2, :] = wp.to_torch(self.gear_assets["gear_large"].data.root_link_quat_w)

        # Select gear data using advanced indexing
        gear_pos_world = self.all_gear_pos_buffer[self.env_indices, self.gear_type_indices]
        gear_quat_world = self.all_gear_quat_buffer[self.env_indices, self.gear_type_indices]

        # Apply rotation offset
        gear_quat_world = math_utils.quat_mul(gear_quat_world, self.grasp_rot_offset_tensor)

        # Get grasp offsets (vectorized)
        self.gear_grasp_offsets_buffer = self.gear_grasp_offsets_stacked[self.gear_type_indices]

        # Transform grasp offsets to world frame
        gear_grasp_pos_world = gear_pos_world + math_utils.quat_apply(gear_quat_world, self.gear_grasp_offsets_buffer)

        # Compute distances
        distances = torch.linalg.norm(gear_grasp_pos_world - eef_pos_world, dim=-1)

        # Check distance threshold
        self.reset_flags[:] = distances > distance_threshold

        return self.reset_flags


class reset_when_gear_orientation_exceeds_threshold(ManagerTermBase):
    """Check if the gear's orientation relative to the gripper exceeds thresholds.

    This class-based term pre-caches all required tensors and thresholds.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        """Initialize the reset when gear orientation exceeds threshold term.

        Args:
            cfg: Termination term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get robot asset configuration
        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        # Validate required parameters
        if "end_effector_body_name" not in cfg.params:
            raise ValueError(
                "'end_effector_body_name' parameter is required in reset_when_gear_orientation_exceeds_threshold"
                " configuration. Example: 'wrist_3_link'"
            )
        if "grasp_rot_offset" not in cfg.params:
            raise ValueError(
                "'grasp_rot_offset' parameter is required in reset_when_gear_orientation_exceeds_threshold"
                " configuration. It should be a quaternion [x, y, z, w]. Example: [0.707, 0.707, 0.0, 0.0]"
            )

        self.end_effector_body_name = cfg.params["end_effector_body_name"]

        # Pre-cache grasp rotation offset tensor
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        # Pre-allocate buffers
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)
        self.all_gear_quat_buffer = torch.zeros(env.num_envs, 3, 4, device=env.device, dtype=torch.float32)
        self.reset_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

        # Find end effector index once
        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            logger.warning(
                f"{self.end_effector_body_name} not found in robot body names. Cannot check gear orientation condition."
            )
            self.eef_idx = None
        else:
            self.eef_idx = eef_indices[0]

    def __call__(
        self,
        env: ManagerBasedEnv,
        roll_threshold_deg: float = 30.0,
        pitch_threshold_deg: float = 30.0,
        yaw_threshold_deg: float = 180.0,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        end_effector_body_name: str | None = None,
        grasp_rot_offset: list | None = None,
    ) -> torch.Tensor:
        """Check if gear orientation exceeds thresholds and return reset flags.

        Args:
            env: Environment instance
            roll_threshold_deg: Maximum allowed roll angle deviation in degrees
            pitch_threshold_deg: Maximum allowed pitch angle deviation in degrees
            yaw_threshold_deg: Maximum allowed yaw angle deviation in degrees
            robot_asset_cfg: Configuration for the robot asset (unused, kept for compatibility)

        Returns:
            Boolean tensor indicating which environments should be reset
        """
        # Reset flags
        self.reset_flags.fill_(False)

        if self.eef_idx is None:
            return self.reset_flags

        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this termination term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Convert thresholds to radians
        roll_threshold_rad = torch.deg2rad(torch.tensor(roll_threshold_deg, device=env.device))
        pitch_threshold_rad = torch.deg2rad(torch.tensor(pitch_threshold_deg, device=env.device))
        yaw_threshold_rad = torch.deg2rad(torch.tensor(yaw_threshold_deg, device=env.device))

        # Get end effector orientation
        eef_quat_world = wp.to_torch(self.robot_asset.data.body_link_quat_w)[:, self.eef_idx]

        # Update gear quaternions in buffer
        self.all_gear_quat_buffer[:, 0, :] = wp.to_torch(self.gear_assets["gear_small"].data.root_link_quat_w)
        self.all_gear_quat_buffer[:, 1, :] = wp.to_torch(self.gear_assets["gear_medium"].data.root_link_quat_w)
        self.all_gear_quat_buffer[:, 2, :] = wp.to_torch(self.gear_assets["gear_large"].data.root_link_quat_w)

        # Select gear data using advanced indexing
        gear_quat_world = self.all_gear_quat_buffer[self.env_indices, self.gear_type_indices]

        # Apply rotation offset
        gear_quat_world = math_utils.quat_mul(gear_quat_world, self.grasp_rot_offset_tensor)

        # Compute relative orientation: q_rel = q_gear * q_eef^-1
        eef_quat_inv = math_utils.quat_conjugate(eef_quat_world)
        relative_quat = math_utils.quat_mul(gear_quat_world, eef_quat_inv)

        # Convert relative quaternion to Euler angles
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(relative_quat)

        # Check if any angle exceeds its threshold
        self.reset_flags[:] = (
            (torch.abs(roll) > roll_threshold_rad)
            | (torch.abs(pitch) > pitch_threshold_rad)
            | (torch.abs(yaw) > yaw_threshold_rad)
        )

        return self.reset_flags
