# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based reward terms for the gear assembly manipulation environment.

Migrated from PhysX to Newton. Key changes:
- Data access returns Warp arrays -> ``wp.to_torch()`` for torch operations
- ``combine_frame_transforms`` uses XYZW natively on Newton
- Identity quaternion is ``[0, 0, 0, 1]`` in XYZW
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import warp as wp

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .events import randomize_gear_type


class keypoint_entity_error(ManagerTermBase):
    """Compute keypoint distance between a RigidObject and the dynamically selected gear.

    Newton migration: Data access uses ``wp.to_torch()`` on Warp arrays.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg_1: SceneEntityCfg = cfg.params.get("asset_cfg_1", SceneEntityCfg("factory_gear_base"))
        self.asset_1 = env.scene[self.asset_cfg_1.name]

        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_1: SceneEntityCfg,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        # Get current pose of asset_1 (convert from Warp)
        curr_pos_1 = wp.to_torch(self.asset_1.data.body_link_pos_w)[:, 0]
        curr_quat_1 = wp.to_torch(self.asset_1.data.body_link_quat_w)[:, 0]

        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this reward term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions and quaternions (convert from Warp)
        all_gear_pos = torch.stack(
            [
                wp.to_torch(self.gear_assets["gear_small"].data.body_link_pos_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_medium"].data.body_link_pos_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_large"].data.body_link_pos_w)[:, 0],
            ],
            dim=1,
        )

        all_gear_quat = torch.stack(
            [
                wp.to_torch(self.gear_assets["gear_small"].data.body_link_quat_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_medium"].data.body_link_quat_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_large"].data.body_link_quat_w)[:, 0],
            ],
            dim=1,
        )

        curr_pos_2 = all_gear_pos[self.env_indices, self.gear_type_indices]
        curr_quat_2 = all_gear_quat[self.env_indices, self.gear_type_indices]

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

    Newton migration: Data access uses ``wp.to_torch()`` on Warp arrays.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg_1: SceneEntityCfg = cfg.params.get("asset_cfg_1", SceneEntityCfg("factory_gear_base"))
        self.asset_1 = env.scene[self.asset_cfg_1.name]

        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }
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
        # Get current pose of asset_1 (convert from Warp)
        curr_pos_1 = wp.to_torch(self.asset_1.data.body_link_pos_w)[:, 0]
        curr_quat_1 = wp.to_torch(self.asset_1.data.body_link_quat_w)[:, 0]

        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this reward term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions and quaternions (convert from Warp)
        all_gear_pos = torch.stack(
            [
                wp.to_torch(self.gear_assets["gear_small"].data.body_link_pos_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_medium"].data.body_link_pos_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_large"].data.body_link_pos_w)[:, 0],
            ],
            dim=1,
        )

        all_gear_quat = torch.stack(
            [
                wp.to_torch(self.gear_assets["gear_small"].data.body_link_quat_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_medium"].data.body_link_quat_w)[:, 0],
                wp.to_torch(self.gear_assets["gear_large"].data.body_link_quat_w)[:, 0],
            ],
            dim=1,
        )

        curr_pos_2 = all_gear_pos[self.env_indices, self.gear_type_indices]
        curr_quat_2 = all_gear_quat[self.env_indices, self.gear_type_indices]

        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
        )

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
    """Get keypoints for pose alignment comparison."""
    if add_cube_center_kp:
        keypoint_corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        keypoint_corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    keypoint_corners = torch.tensor(keypoint_corners, device=device, dtype=torch.float32)
    keypoint_corners = torch.cat((keypoint_corners, -keypoint_corners[-3:]), dim=0)

    return keypoint_corners


class _compute_keypoint_distance:
    """Compute keypoint distance between current and target poses.

    Identity quaternion uses XYZW convention: [0, 0, 0, 1].
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        add_cube_center_kp = cfg.params.get("add_cube_center_kp", True)

        self.keypoint_offsets_base = _get_keypoint_offsets_full_6d(
            add_cube_center_kp=add_cube_center_kp, device=env.device
        )
        self.num_keypoints = self.keypoint_offsets_base.shape[0]

        # XYZW identity quaternion: [0, 0, 0, 1]
        self.identity_quat_keypoints = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs * self.num_keypoints, 1)
            .contiguous()
        )

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
        """Compute keypoint distance between current and target poses."""
        num_envs = current_pos.shape[0]
        keypoint_offsets = self.keypoint_offsets_base * keypoint_scale

        self.keypoint_offsets_buffer[:num_envs] = keypoint_offsets.unsqueeze(0)

        keypoint_offsets_flat = self.keypoint_offsets_buffer[:num_envs].reshape(-1, 3)
        identity_quat = self.identity_quat_keypoints[: num_envs * self.num_keypoints]

        current_quat_expanded = current_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4)
        current_pos_expanded = current_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 3)
        target_quat_expanded = target_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4)
        target_pos_expanded = target_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 3)

        keypoints_current_flat, _ = combine_frame_transforms(
            current_pos_expanded, current_quat_expanded, keypoint_offsets_flat, identity_quat
        )
        keypoints_target_flat, _ = combine_frame_transforms(
            target_pos_expanded, target_quat_expanded, keypoint_offsets_flat, identity_quat
        )

        keypoints_current = keypoints_current_flat.reshape(num_envs, self.num_keypoints, 3)
        keypoints_target = keypoints_target_flat.reshape(num_envs, self.num_keypoints, 3)

        keypoint_dist_sep = torch.norm(keypoints_target - keypoints_current, p=2, dim=-1)
        return keypoint_dist_sep
