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

if TYPE_CHECKING:
    from isaaclab_experimental.managers.manager_term_cfg import RewardTermCfg

    from isaaclab.assets import Articulation
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
    """Write 1.0 when the body-frame up projection exceeds the threshold, else 0.0, into the reward output."""
    i = wp.tid()
    up_proj = -rotate_vec_to_body_frame(wp.normalize(gravity_w[i]), root_pose_w[i])[2]
    out[i] = wp.where(up_proj > threshold, 1.0, 0.0)


def upright_posture_bonus(
    env: ManagerBasedRLEnv, out, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reward for maintaining an upright posture. Writes 1.0 if up_proj > threshold, else 0.0."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_upright_posture_bonus_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_link_pose_w.warp, asset.data.GRAVITY_VEC_W.warp, threshold, out],
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
    """Write the heading-to-target bonus (1.0 above threshold, else the heading projection ratio)."""
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
        inputs=[
            asset.data.root_pos_w.warp,
            asset.data.root_quat_w.warp,
            target_pos[0],
            target_pos[1],
            threshold,
            out,
        ],
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
    """Initialize masked envs' potential from the negative 3D distance to the target scaled by the inverse step time."""
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
    """Update the potential from the xy distance to the target and write its per-step increase as the reward."""
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
                asset.data.root_pos_w.warp,
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
            inputs=[asset.data.root_pos_w.warp, target_pos[0], target_pos[1], inv_dt, self.potentials, out],
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
    """Accumulate the gear-ratio-weighted joint-position-limit violation beyond the threshold."""
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
                asset.data.joint_pos.warp,
                asset.data.soft_joint_pos_limits.warp,
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
    """Accumulate the gear-scaled absolute action-times-joint-velocity power over joints into the penalty output."""
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
            inputs=[env.action_manager.action, asset.data.joint_vel.warp, self._gear_ratio_scaled_wp, out],
            device=env.device,
        )


# ---------------------------------------------------------------------------
# terminated_penalty
# ---------------------------------------------------------------------------


@wp.kernel
def _terminated_penalty_kernel(
    terminated: wp.array(dtype=wp.bool),
    inv_dt: float,
    out: wp.array(dtype=wp.float32),
):
    """Emit ``1 / step_dt`` on the terminating step so the manager's dt scaling cancels."""
    i = wp.tid()
    out[i] = wp.where(terminated[i], inv_dt, 0.0)


def terminated_penalty(env: ManagerBasedRLEnv, out: wp.array(dtype=wp.float32)) -> None:
    """One-off penalty for terminating early, independent of the environment step size.

    Warp-first override of :func:`isaaclab_tasks.core.locomotion.mdp.rewards.terminated_penalty`.
    """
    wp.launch(
        kernel=_terminated_penalty_kernel,
        dim=env.num_envs,
        inputs=[env.termination_manager.terminated_wp, 1.0 / env.step_dt, out],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# survival_success_rate
# ---------------------------------------------------------------------------


@wp.kernel
def _survival_counts_kernel(
    env_mask: wp.array(dtype=wp.bool),
    time_outs: wp.array(dtype=wp.bool),
    counts: wp.array(dtype=wp.int32),
):
    """Atomically count masked (just-reset) envs and how many of them timed out."""
    env_index = wp.tid()
    if env_mask[env_index]:
        wp.atomic_add(counts, 0, 1)
        if time_outs[env_index]:
            wp.atomic_add(counts, 1, 1)


@wp.kernel
def _survival_rate_kernel(
    counts: wp.array(dtype=wp.int32),
    success_rate: wp.array(dtype=wp.float32),
):
    """Compute the survival success rate as the fraction of just-reset envs that timed out.

    Left undivided when no env reset: 0/0 is NaN, which is what the stable term's mean over an
    empty selection yields. Substituting 0.0 would log a real 0% sample for a reset that
    selected nothing.
    """
    success_rate[0] = wp.float32(counts[1]) / wp.float32(counts[0])


class survival_success_rate(ManagerTermBase):
    """Tracks episode survival as the success metric (Warp-first).

    Twin of :class:`isaaclab_tasks.core.locomotion.mdp.rewards.survival_success_rate`.
    Returns zero reward (pure metric tracking). On reset, computes the fraction of
    just-reset environments that timed out (survived the full episode) entirely on-device
    and exposes it as ``Metrics/success_rate`` through the reward manager's reset extras.
    Unlike the stable term, there is no host readback, so the computation stays
    CUDA-graph capturable.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # [0] number of just-reset envs, [1] number of those that timed out
        self._counts_wp = wp.zeros((2,), dtype=wp.int32, device=env.device)
        self._success_rate_wp = wp.zeros((1,), dtype=wp.float32, device=env.device)
        # persistent 0-dim tensor view: kernels refresh the value on every reset/replay
        self._reset_extras = {"Metrics/success_rate": wp.to_torch(self._success_rate_wp)[0]}

    def reset(self, env_mask: wp.array | None = None) -> dict[str, torch.Tensor]:
        if env_mask is None:
            env_mask = self._env.resolve_env_mask()
        self._counts_wp.zero_()
        wp.launch(
            kernel=_survival_counts_kernel,
            dim=self.num_envs,
            inputs=[env_mask, self._env.termination_manager.time_outs_wp, self._counts_wp],
            device=self.device,
        )
        wp.launch(
            kernel=_survival_rate_kernel,
            dim=1,
            inputs=[self._counts_wp, self._success_rate_wp],
            device=self.device,
        )
        return self._reset_extras

    def __call__(self, env: ManagerBasedRLEnv, out: wp.array(dtype=wp.float32)) -> None:
        out.zero_()
