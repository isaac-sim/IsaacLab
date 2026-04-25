# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first reward terms for the humanoid task.

All reward functions follow the ``func(env, out, **params) -> None`` signature
where ``out`` is a pre-allocated Warp array of shape ``(num_envs,)`` with float32 dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.managers.manager_base import ManagerTermBase
from isaaclab_newton.kernels.state_kernels import rotate_vec_to_body_frame

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab_experimental.managers.manager_term_cfg import RewardTermCfg

    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Function-based reward terms
# ---------------------------------------------------------------------------


# Inline Tier 1 access: derives projected gravity directly from root_link_pose_w,
# avoiding the lazy TimestampedWarpBuffer which is not CUDA-graph-capturable.
# See GRAPH_CAPTURE_MIGRATION.md in isaaclab_newton for background.
# If ArticulationData Tier 2 lazy update is made graph-safe, this can revert to
# reading asset.data.projected_gravity_b directly.


@wp.kernel
def _upright_posture_bonus_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    gravity_w: wp.array(dtype=wp.vec3f),
    threshold: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    up_proj = -rotate_vec_to_body_frame(gravity_w[0], root_pose_w[i])[2]
    out[i] = wp.where(up_proj > threshold, 1.0, 0.0)


def upright_posture_bonus(
    env: ManagerBasedRLEnv, out, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reward for maintaining an upright posture. Writes 1.0 if up_proj > threshold, else 0.0."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_upright_posture_bonus_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_link_pose_w, asset.data.GRAVITY_VEC_W, threshold, out],
        device=env.device,
    )


@wp.kernel
def _move_to_target_bonus_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    target_x: float,
    target_y: float,
    threshold: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    pos = root_pos_w[i]
    q = root_quat_w[i]
    # direction to target
    dx = target_x - pos[0]
    dy = target_y - pos[1]
    dist = wp.sqrt(dx * dx + dy * dy)
    inv_dist = wp.where(dist > 1.0e-6, 1.0 / dist, 0.0)
    to_target_x = dx * inv_dist
    to_target_y = dy * inv_dist
    # forward vector
    fwd = wp.quat_rotate(q, wp.vec3f(1.0, 0.0, 0.0))
    heading_proj = fwd[0] * to_target_x + fwd[1] * to_target_y
    out[i] = wp.where(heading_proj > threshold, 1.0, heading_proj / threshold)


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    out,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reward for heading towards the target."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_move_to_target_bonus_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w, asset.data.root_quat_w, target_pos[0], target_pos[1], threshold, out],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# Class-based reward terms
# ---------------------------------------------------------------------------


@wp.kernel
def _progress_reward_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    root_pos_w: wp.array(dtype=wp.vec3f),
    target_x: float,
    target_y: float,
    target_z: float,
    inv_step_dt: float,
    potentials: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if env_mask[i]:
        pos = root_pos_w[i]
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        dz = target_z - pos[2]
        dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
        potentials[i] = -dist * inv_step_dt


@wp.kernel
def _progress_reward_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    target_x: float,
    target_y: float,
    inv_step_dt: float,
    potentials: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    pos = root_pos_w[i]
    dx = target_x - pos[0]
    dy = target_y - pos[1]
    # z component is zeroed (xy distance only, matching stable)
    dist = wp.sqrt(dx * dx + dy * dy)
    prev = potentials[i]
    pot = -dist * inv_step_dt
    potentials[i] = pot
    out[i] = pot - prev


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target (potential-based)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.potentials = wp.zeros(env.num_envs, dtype=wp.float32, device=env.device)
        self._target_pos = cfg.params["target_pos"]

    def reset(self, env_mask: wp.array | None = None) -> None:
        if env_mask is None:
            self.potentials.zero_()
            return
        asset: Articulation = self._env.scene["robot"]
        inv_dt = 1.0 / self._env.step_dt
        wp.launch(
            kernel=_progress_reward_reset_kernel,
            dim=self.num_envs,
            inputs=[
                env_mask,
                asset.data.root_pos_w,
                self._target_pos[0],
                self._target_pos[1],
                self._target_pos[2],
                inv_dt,
                self.potentials,
            ],
            device=self.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        out,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        asset: Articulation = env.scene[asset_cfg.name]
        inv_dt = 1.0 / env.step_dt
        wp.launch(
            kernel=_progress_reward_kernel,
            dim=env.num_envs,
            inputs=[asset.data.root_pos_w, target_pos[0], target_pos[1], inv_dt, self.potentials, out],
            device=env.device,
        )


@wp.kernel
def _joint_pos_limits_penalty_ratio_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    soft_limits: wp.array(dtype=wp.vec2f, ndim=2),
    gear_ratio_scaled: wp.array(dtype=wp.float32, ndim=2),
    threshold: float,
    inv_range: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n_joints = joint_pos.shape[1]
    s = float(0.0)
    for j in range(n_joints):
        lim = soft_limits[i, j]
        lower = lim.x
        upper = lim.y
        mid = (lower + upper) * 0.5
        half_range = (upper - lower) * 0.5
        scaled = float(0.0)
        if half_range > 0.0:
            scaled = (joint_pos[i, j] - mid) / half_range
        abs_scaled = wp.abs(scaled)
        if abs_scaled > threshold:
            violation = (abs_scaled - threshold) * inv_range
            s += violation * gear_ratio_scaled[i, j]
    out[i] = s


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint (torch in __init__ is fine)
        gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        gear_ratio_scaled = gear_ratio / torch.max(gear_ratio)
        self._gear_ratio_scaled_wp = wp.from_torch(gear_ratio_scaled)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        out,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        asset: Articulation = env.scene[asset_cfg.name]
        wp.launch(
            kernel=_joint_pos_limits_penalty_ratio_kernel,
            dim=env.num_envs,
            inputs=[
                asset.data.joint_pos,
                asset.data.soft_joint_pos_limits,
                self._gear_ratio_scaled_wp,
                threshold,
                1.0 / (1.0 - threshold),
                out,
            ],
            device=env.device,
        )


@wp.kernel
def _power_consumption_kernel(
    action: wp.array(dtype=wp.float32, ndim=2),
    joint_vel: wp.array(dtype=wp.float32, ndim=2),
    gear_ratio_scaled: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n_joints = action.shape[1]
    s = float(0.0)
    for j in range(n_joints):
        s += wp.abs(action[i, j] * joint_vel[i, j] * gear_ratio_scaled[i, j])
    out[i] = s


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint (torch in __init__ is fine)
        gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        gear_ratio_scaled = gear_ratio / torch.max(gear_ratio)
        self._gear_ratio_scaled_wp = wp.from_torch(gear_ratio_scaled)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        out,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        asset: Articulation = env.scene[asset_cfg.name]
        wp.launch(
            kernel=_power_consumption_kernel,
            dim=env.num_envs,
            inputs=[env.action_manager.action, asset.data.joint_vel, self._gear_ratio_scaled_wp, out],
            device=env.device,
        )
