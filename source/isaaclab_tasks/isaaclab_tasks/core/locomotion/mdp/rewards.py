# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the locomotion (ant and humanoid) environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute the planar distance to the target, matching __call__ so the first step scores no progress
        to_target_pos = obs.walk_target_w(self._env, self.cfg.params["target_pos"])[env_ids]
        to_target_pos = to_target_pos - asset.data.root_pos_w.torch[env_ids, :3]
        to_target_pos[:, 2] = 0.0
        # reward terms
        self.potentials[env_ids] = -torch.linalg.norm(to_target_pos, ord=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        to_target_pos = obs.walk_target_w(env, target_pos) - asset.data.root_pos_w.torch[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.linalg.norm(to_target_pos, ord=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio_scaled = _resolve_scaled_gear_ratio(cfg.params["gear_ratio"], asset, env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos.torch,
            asset.data.soft_joint_pos_limits.torch[..., 0],
            asset.data.soft_joint_pos_limits.torch[..., 1],
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the joint actions.

    Computed as the action scaled by its gear ratio, normalized by the largest gear ratio, times the joint
    velocity, summed over joints. The effort action terms scale the actions by the same gear ratios, so up
    to that normalization and the effort clip this is the commanded effort times the joint velocity. It
    matches the electricity cost of the direct locomotion environments.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio_scaled = _resolve_scaled_gear_ratio(cfg.params["gear_ratio"], asset, env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # power = effort * velocity, with the effort taken as the gear-normalized action
        return torch.sum(
            torch.abs(env.action_manager.action * asset.data.joint_vel.torch * self.gear_ratio_scaled), dim=-1
        )


def _resolve_scaled_gear_ratio(gear_ratio: dict[str, float], asset: Articulation, device: str) -> torch.Tensor:
    """Resolve the per-joint gear ratios and normalize them by the largest one.

    Joints that the ``gear_ratio`` table does not match keep a unit gear.

    Args:
        gear_ratio: Gear ratio per joint name expression.
        asset: Articulation whose joint names the expressions are matched against.
        device: Device of the returned tensor.

    Returns:
        Gear ratios divided by the maximum gear ratio, shape ``(num_joints,)``. Broadcasts against the
        ``(num_envs, num_joints)`` joint tensors.
    """
    gears = torch.ones(asset.num_joints, device=device)
    joint_ids, _, values = string_utils.resolve_matching_names_values(gear_ratio, asset.joint_names)
    gears[joint_ids] = torch.tensor(values, device=device)
    return gears / torch.max(gears)
