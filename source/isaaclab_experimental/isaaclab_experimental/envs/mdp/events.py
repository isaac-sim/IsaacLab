# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first overrides for common event terms.

These functions are intended to be used with the experimental Warp-first
:class:`isaaclab_experimental.managers.EventManager` (mask-based interval/reset).

Why this exists:
- Stable event terms (e.g. `isaaclab.envs.mdp.events.reset_joints_by_offset`) often build torch tensors and then
  call into Newton articulation writers with partial indices (env_ids/joint_ids).
- On the Newton backend, passing torch tensors triggers expensive torch->warp conversions that currently allocate
  full `(num_envs, num_joints)` buffers (see `isaaclab.utils.warp.utils.make_complete_data_from_torch_dual_index`).

These Warp-first implementations avoid that by writing directly into the sim-bound Warp state buffers
(`asset.data.joint_pos` / `asset.data.joint_vel`) for the selected envs/joints.

Notes:
- These terms assume the Newton/Warp backend (Warp arrays are available for joint state and defaults).
- For best performance, pass :class:`isaaclab_experimental.managers.SceneEntityCfg` so `joint_ids_wp` is cached.
"""

from __future__ import annotations

import warp as wp

from isaaclab.assets import Articulation

from isaaclab_experimental.managers import SceneEntityCfg


@wp.kernel
def _reset_joints_by_offset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    joint_ids: wp.array(dtype=wp.int32),
    rng_state: wp.array(dtype=wp.uint32),
    default_joint_pos: wp.array(dtype=wp.float32, ndim=2),
    default_joint_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_vel: wp.array(dtype=wp.float32, ndim=2),
    soft_joint_pos_limits: wp.array(dtype=wp.vec2f, ndim=2),
    soft_joint_vel_limits: wp.array(dtype=wp.float32, ndim=2),
    pos_lo: float,
    pos_hi: float,
    vel_lo: float,
    vel_hi: float,
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    # 1 thread per env so per-env RNG state updates are race-free.
    state = rng_state[env_id]
    for joint_i in range(joint_ids.shape[0]):
        joint_id = joint_ids[joint_i]

        # offset samples in the provided ranges (Warp RNG state pattern)
        pos_off = wp.randf(state, pos_lo, pos_hi)
        vel_off = wp.randf(state, vel_lo, vel_hi)

        pos = default_joint_pos[env_id, joint_id] + pos_off
        vel = default_joint_vel[env_id, joint_id] + vel_off

        # clamp to soft limits
        lim = soft_joint_pos_limits[env_id, joint_id]
        pos = wp.clamp(pos, lim.x, lim.y)
        vmax = soft_joint_vel_limits[env_id, joint_id]
        vel = wp.clamp(vel, -vmax, vmax)

        # write into sim-bound state buffers
        joint_pos[env_id, joint_id] = pos
        joint_vel[env_id, joint_id] = vel

    rng_state[env_id] = state


def reset_joints_by_offset(
    env,
    env_mask: wp.array,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Warp-first reset of joint state by random offsets around defaults.

    This overrides the stable `isaaclab.envs.mdp.events.reset_joints_by_offset` when importing
    via `isaaclab_experimental.envs.mdp`.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Assume cfg params are already resolved by the manager stack (Warp-first workflow).
    if asset_cfg.joint_ids_wp is None:
        raise ValueError(
            f"reset_joints_by_offset requires an experimental SceneEntityCfg with resolved joint_ids_wp, "
            f"but got None for asset '{asset_cfg.name}'. "
            "Use isaaclab_experimental.managers.SceneEntityCfg and ensure joint_names are set."
        )
    if not hasattr(env, "rng_state_wp") or env.rng_state_wp is None:
        raise AttributeError(
            "reset_joints_by_offset requires env.rng_state_wp to be initialized. "
            "Use ManagerBasedEnvWarp or ManagerBasedRLEnvWarp as the base environment."
        )

    wp.launch(
        kernel=_reset_joints_by_offset_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            asset_cfg.joint_ids_wp,
            env.rng_state_wp,
            asset.data.default_joint_pos,
            asset.data.default_joint_vel,
            asset.data.joint_pos,
            asset.data.joint_vel,
            asset.data.soft_joint_pos_limits,
            asset.data.soft_joint_vel_limits,
            float(position_range[0]),
            float(position_range[1]),
            float(velocity_range[0]),
            float(velocity_range[1]),
        ],
        device=env.device,
    )


@wp.kernel
def _reset_joints_by_scale_kernel(
    env_mask: wp.array(dtype=wp.bool),
    joint_ids: wp.array(dtype=wp.int32),
    rng_state: wp.array(dtype=wp.uint32),
    default_joint_pos: wp.array(dtype=wp.float32, ndim=2),
    default_joint_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_vel: wp.array(dtype=wp.float32, ndim=2),
    soft_joint_pos_limits: wp.array(dtype=wp.vec2f, ndim=2),
    soft_joint_vel_limits: wp.array(dtype=wp.float32, ndim=2),
    pos_lo: float,
    pos_hi: float,
    vel_lo: float,
    vel_hi: float,
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    state = rng_state[env_id]
    for joint_i in range(joint_ids.shape[0]):
        joint_id = joint_ids[joint_i]

        # scale samples in the provided ranges
        pos_scale = wp.randf(state, pos_lo, pos_hi)
        vel_scale = wp.randf(state, vel_lo, vel_hi)

        pos = default_joint_pos[env_id, joint_id] * pos_scale
        vel = default_joint_vel[env_id, joint_id] * vel_scale

        lim = soft_joint_pos_limits[env_id, joint_id]
        pos = wp.clamp(pos, lim.x, lim.y)
        vmax = soft_joint_vel_limits[env_id, joint_id]
        vel = wp.clamp(vel, -vmax, vmax)

        # write into sim
        joint_pos[env_id, joint_id] = pos
        joint_vel[env_id, joint_id] = vel

    rng_state[env_id] = state


def reset_joints_by_scale(
    env,
    env_mask: wp.array,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Warp-first reset of joint state by scaling defaults with random factors."""
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids_wp is None:
        raise ValueError(
            f"reset_joints_by_scale requires an experimental SceneEntityCfg with resolved joint_ids_wp, "
            f"but got None for asset '{asset_cfg.name}'. "
            "Use isaaclab_experimental.managers.SceneEntityCfg and ensure joint_names are set."
        )
    if not hasattr(env, "rng_state_wp") or env.rng_state_wp is None:
        raise AttributeError(
            "reset_joints_by_scale requires env.rng_state_wp to be initialized. "
            "Use ManagerBasedEnvWarp or ManagerBasedRLEnvWarp as the base environment."
        )

    wp.launch(
        kernel=_reset_joints_by_scale_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            asset_cfg.joint_ids_wp,
            env.rng_state_wp,
            asset.data.default_joint_pos,
            asset.data.default_joint_vel,
            asset.data.joint_pos,
            asset.data.joint_vel,
            asset.data.soft_joint_pos_limits,
            asset.data.soft_joint_vel_limits,
            float(position_range[0]),
            float(position_range[1]),
            float(velocity_range[0]),
            float(velocity_range[1]),
        ],
        device=env.device,
    )
