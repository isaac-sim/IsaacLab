# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based reward terms for the gear assembly manipulation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .events import randomize_gear_type


class keypoint_command_error(ManagerTermBase):
    """Compute keypoint distance between current and desired poses from command.

    This class-based term uses _compute_keypoint_distance internally.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint command error term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("ee_frame"))
        self.command_name: str = cfg.params.get("command_name", "ee_pose")

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute keypoint distance error.

        Args:
            env: Environment instance
            command_name: Name of the command containing desired pose
            asset_cfg: Configuration of the asset to track
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Mean keypoint distance tensor of shape (num_envs,)
        """
        # Extract frame transformer sensor
        asset: FrameTransformer = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        # Get desired pose from command
        des_pos_w = command[:, :3]
        des_quat_w = command[:, 3:7]

        # Get current pose from frame transformer
        curr_pos_w = asset.data.target_pos_source[:, 0]
        curr_quat_w = asset.data.target_quat_source[:, 0]

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_w,
            current_quat=curr_quat_w,
            target_pos=des_pos_w,
            target_quat=des_quat_w,
            keypoint_scale=keypoint_scale,
        )

        return keypoint_dist_sep.mean(-1)


class keypoint_command_error_exp(ManagerTermBase):
    """Compute exponential keypoint reward between current and desired poses from command.

    This class-based term uses _compute_keypoint_distance internally and applies
    exponential reward transformation.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint command error exponential term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("ee_frame"))
        self.command_name: str = cfg.params.get("command_name", "ee_pose")

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute exponential keypoint reward.

        Args:
            env: Environment instance
            command_name: Name of the command containing desired pose
            asset_cfg: Configuration of the asset to track
            kp_exp_coeffs: List of (a, b) coefficient pairs for exponential reward
            kp_use_sum_of_exps: Whether to use sum of exponentials
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Exponential keypoint reward tensor of shape (num_envs,)
        """
        # Extract frame transformer sensor
        asset: FrameTransformer = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        # Get desired pose from command
        des_pos_w = command[:, :3]
        des_quat_w = command[:, 3:7]

        # Get current pose from frame transformer
        curr_pos_w = asset.data.target_pos_source[:, 0]
        curr_quat_w = asset.data.target_quat_source[:, 0]

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_w,
            current_quat=curr_quat_w,
            target_pos=des_pos_w,
            target_quat=des_quat_w,
            keypoint_scale=keypoint_scale,
        )

        # Compute exponential reward
        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])

        if kp_use_sum_of_exps:
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (
                    1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
                ).mean(-1)
        else:
            keypoint_dist = keypoint_dist_sep.mean(-1)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1.0 / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))

        return keypoint_reward_exp


class keypoint_entity_error(ManagerTermBase):
    """Compute keypoint distance between a RigidObject and the dynamically selected gear.

    This class-based term pre-caches gear type mapping and asset references.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint entity error term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg_1: SceneEntityCfg = cfg.params.get("asset_cfg_1", SceneEntityCfg("factory_gear_base"))
        self.asset_1 = env.scene[self.asset_cfg_1.name]

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_1: SceneEntityCfg,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute keypoint distance error.

        Args:
            env: Environment instance
            asset_cfg_1: Configuration of the first asset (RigidObject)
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Mean keypoint distance tensor of shape (num_envs,)
        """
        # Get current pose of asset_1 (RigidObject)
        curr_pos_1 = self.asset_1.data.body_pos_w[:, 0]
        curr_quat_1 = self.asset_1.data.body_quat_w[:, 0]

        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this reward term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions and quaternions
        all_gear_pos = torch.stack(
            [
                self.gear_assets["gear_small"].data.body_pos_w[:, 0],
                self.gear_assets["gear_medium"].data.body_pos_w[:, 0],
                self.gear_assets["gear_large"].data.body_pos_w[:, 0],
            ],
            dim=1,
        )

        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.body_quat_w[:, 0],
                self.gear_assets["gear_medium"].data.body_quat_w[:, 0],
                self.gear_assets["gear_large"].data.body_quat_w[:, 0],
            ],
            dim=1,
        )

        # Select positions and quaternions using advanced indexing
        curr_pos_2 = all_gear_pos[self.env_indices, self.gear_type_indices]
        curr_quat_2 = all_gear_quat[self.env_indices, self.gear_type_indices]

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
        )

        return keypoint_dist_sep.mean(-1)


class keypoint_entity_error_exp(ManagerTermBase):
    """Compute exponential keypoint reward between a RigidObject and the dynamically selected gear.

    This class-based term pre-caches gear type mapping and asset references.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint entity error exponential term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg_1: SceneEntityCfg = cfg.params.get("asset_cfg_1", SceneEntityCfg("factory_gear_base"))
        self.asset_1 = env.scene[self.asset_cfg_1.name]

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_1: SceneEntityCfg,
        kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute exponential keypoint reward.

        Args:
            env: Environment instance
            asset_cfg_1: Configuration of the first asset (RigidObject)
            kp_exp_coeffs: List of (a, b) coefficient pairs for exponential reward
            kp_use_sum_of_exps: Whether to use sum of exponentials
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Exponential keypoint reward tensor of shape (num_envs,)
        """
        # Get current pose of asset_1 (RigidObject)
        curr_pos_1 = self.asset_1.data.body_pos_w[:, 0]
        curr_quat_1 = self.asset_1.data.body_quat_w[:, 0]

        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this reward term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions and quaternions
        all_gear_pos = torch.stack(
            [
                self.gear_assets["gear_small"].data.body_pos_w[:, 0],
                self.gear_assets["gear_medium"].data.body_pos_w[:, 0],
                self.gear_assets["gear_large"].data.body_pos_w[:, 0],
            ],
            dim=1,
        )

        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.body_quat_w[:, 0],
                self.gear_assets["gear_medium"].data.body_quat_w[:, 0],
                self.gear_assets["gear_large"].data.body_quat_w[:, 0],
            ],
            dim=1,
        )

        # Select positions and quaternions using advanced indexing
        curr_pos_2 = all_gear_pos[self.env_indices, self.gear_type_indices]
        curr_quat_2 = all_gear_quat[self.env_indices, self.gear_type_indices]

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
        )

        # Compute exponential reward
        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])

        if kp_use_sum_of_exps:
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (
                    1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
                ).mean(-1)
        else:
            keypoint_dist = keypoint_dist_sep.mean(-1)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1.0 / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))

        return keypoint_reward_exp


##
# Helper functions and classes
##


def _get_keypoint_offsets_full_6d(add_cube_center_kp: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Get keypoints for pose alignment comparison. Pose is aligned if all axis are aligned.

    Args:
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)
        device: Device to create the tensor on

    Returns:
        Keypoint offsets tensor of shape (num_keypoints, 3)
    """
    if add_cube_center_kp:
        keypoint_corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        keypoint_corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    keypoint_corners = torch.tensor(keypoint_corners, device=device, dtype=torch.float32)
    keypoint_corners = torch.cat((keypoint_corners, -keypoint_corners[-3:]), dim=0)

    return keypoint_corners


class _compute_keypoint_distance:
    """Compute keypoint distance between current and target poses.

    This helper class pre-caches keypoint offsets and identity quaternions
    to avoid repeated allocations during reward computation.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the compute keypoint distance helper.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        # Get keypoint configuration
        add_cube_center_kp = cfg.params.get("add_cube_center_kp", True)

        # Pre-compute base keypoint offsets (unscaled)
        self.keypoint_offsets_base = _get_keypoint_offsets_full_6d(
            add_cube_center_kp=add_cube_center_kp, device=env.device
        )
        self.num_keypoints = self.keypoint_offsets_base.shape[0]

        # Pre-allocate identity quaternion for keypoint transforms
        self.identity_quat_keypoints = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs * self.num_keypoints, 1)
            .contiguous()
        )

        # Pre-allocate buffer for batched keypoint offsets
        self.keypoint_offsets_buffer = torch.zeros(
            env.num_envs, self.num_keypoints, 3, device=env.device, dtype=torch.float32
        )

    def compute(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        keypoint_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute keypoint distance between current and target poses.

        Args:
            current_pos: Current position tensor of shape (num_envs, 3)
            current_quat: Current quaternion tensor of shape (num_envs, 4)
            target_pos: Target position tensor of shape (num_envs, 3)
            target_quat: Target quaternion tensor of shape (num_envs, 4)
            keypoint_scale: Scale factor for keypoint offsets

        Returns:
            Keypoint distance tensor of shape (num_envs, num_keypoints)
        """
        num_envs = current_pos.shape[0]

        # Scale keypoint offsets
        keypoint_offsets = self.keypoint_offsets_base * keypoint_scale

        # Create batched keypoints (in-place operation)
        self.keypoint_offsets_buffer[:num_envs] = keypoint_offsets.unsqueeze(0)

        # Flatten for batch processing
        keypoint_offsets_flat = self.keypoint_offsets_buffer[:num_envs].reshape(-1, 3)
        identity_quat = self.identity_quat_keypoints[: num_envs * self.num_keypoints]

        # Expand quaternions and positions for all keypoints
        current_quat_expanded = current_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4)
        current_pos_expanded = current_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 3)
        target_quat_expanded = target_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4)
        target_pos_expanded = target_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 3)

        # Transform all keypoints at once
        keypoints_current_flat, _ = combine_frame_transforms(
            current_pos_expanded, current_quat_expanded, keypoint_offsets_flat, identity_quat
        )
        keypoints_target_flat, _ = combine_frame_transforms(
            target_pos_expanded, target_quat_expanded, keypoint_offsets_flat, identity_quat
        )

        # Reshape back
        keypoints_current = keypoints_current_flat.reshape(num_envs, self.num_keypoints, 3)
        keypoints_target = keypoints_target_flat.reshape(num_envs, self.num_keypoints, 3)

        # Calculate L2 norm distance
        keypoint_dist_sep = torch.norm(keypoints_target - keypoints_current, p=2, dim=-1)

        return keypoint_dist_sep
