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
  full `(num_envs, num_joints)` buffers.

These Warp-first implementations avoid that by writing directly into the sim-bound Warp state buffers
(`asset.data.joint_pos` / `asset.data.joint_vel`) for the selected envs/joints.

Notes:
- These terms assume the Newton/Warp backend (Warp arrays are available for joint state and defaults).
- For best performance, pass :class:`isaaclab_experimental.managers.SceneEntityCfg` so `joint_ids_wp` is cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import warp as wp

from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.utils.warp import WarpCapturable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _range_cache_key(
    range_values: dict[str, tuple[float, float]] | tuple[float, float],
) -> tuple[object, ...]:
    """Return an immutable snapshot of an event range for state-cache lookup."""
    if isinstance(range_values, dict):
        return tuple((name, tuple(bounds)) for name, bounds in sorted(range_values.items()))
    return tuple(range_values)


class _EventStateCache:
    """Environment-owned storage for persistent Warp event buffers and constants.

    The cache lifetime matches the environment lifetime. Keys include the event
    callable, asset, term configuration, and immutable range values so stale parsed
    constants are not reused after a range changes.
    """

    _ENV_ATTRIBUTE: ClassVar[str] = "_warp_event_state_cache"

    @classmethod
    def for_env(cls, env: ManagerBasedEnv) -> dict[tuple[object, ...], object]:
        """Return the event-state cache owned by an environment."""
        cache = getattr(env, cls._ENV_ATTRIBUTE, None)
        if cache is None:
            cache = {}
            setattr(env, cls._ENV_ATTRIBUTE, cache)
        return cache


class _RandomizeRigidBodyComState:
    """Persistent arguments for center-of-mass randomization."""

    def __init__(
        self,
        asset: RigidObject | Articulation,
        asset_cfg: SceneEntityCfg,
        com_range: dict[str, tuple[float, float]],
    ):
        self.asset = asset
        self.asset_cfg = asset_cfg
        self.com_range = com_range
        self.default_com = wp.clone(asset.data.body_com_pos_b.warp)
        ranges = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
        self.com_lo = wp.vec3f(ranges[0][0], ranges[1][0], ranges[2][0])
        self.com_hi = wp.vec3f(ranges[0][1], ranges[1][1], ranges[2][1])


class _ApplyExternalForceTorqueState:
    """Persistent output buffers for external wrench randomization."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        asset: RigidObject | Articulation,
        asset_cfg: SceneEntityCfg,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
    ):
        self.asset = asset
        self.asset_cfg = asset_cfg
        self.force_range = force_range
        self.torque_range = torque_range
        self.forces = wp.zeros((env.num_envs, asset.num_bodies), dtype=wp.vec3f, device=env.device)
        self.torques = wp.zeros((env.num_envs, asset.num_bodies), dtype=wp.vec3f, device=env.device)
        if asset_cfg.body_ids is None or asset_cfg.body_ids == slice(None):
            body_ids = list(range(asset.num_bodies))
        elif isinstance(asset_cfg.body_ids, int):
            body_ids = [asset_cfg.body_ids]
        else:
            body_ids = list(asset_cfg.body_ids)
        body_mask = [False] * asset.num_bodies
        for body_id in body_ids:
            body_mask[body_id] = True
        self.body_ids = wp.array(body_ids, dtype=wp.int32, device=env.device)
        self.body_mask = wp.array(body_mask, dtype=wp.bool, device=env.device)


class _PushBySettingVelocityState:
    """Persistent output buffer and ranges for root-velocity pushes."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        asset: RigidObject | Articulation,
        asset_cfg: SceneEntityCfg,
        velocity_range: dict[str, tuple[float, float]],
    ):
        self.asset = asset
        self.asset_cfg = asset_cfg
        self.velocity_range = velocity_range
        self.velocity = wp.zeros((env.num_envs,), dtype=wp.spatial_vectorf, device=env.device)
        ranges = [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        self.lin_lo = wp.vec3f(ranges[0][0], ranges[1][0], ranges[2][0])
        self.lin_hi = wp.vec3f(ranges[0][1], ranges[1][1], ranges[2][1])
        self.ang_lo = wp.vec3f(ranges[3][0], ranges[4][0], ranges[5][0])
        self.ang_hi = wp.vec3f(ranges[3][1], ranges[4][1], ranges[5][1])


class _ResetRootStateUniformState:
    """Persistent output buffers and ranges for uniform root-state reset."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        asset: RigidObject | Articulation,
        asset_cfg: SceneEntityCfg,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
    ):
        self.asset = asset
        self.asset_cfg = asset_cfg
        self.pose_range = pose_range
        self.velocity_range = velocity_range
        self.pose = wp.zeros((env.num_envs,), dtype=wp.transformf, device=env.device)
        self.velocity = wp.zeros((env.num_envs,), dtype=wp.spatial_vectorf, device=env.device)

        pose_ranges = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        self.pos_lo = wp.vec3f(pose_ranges[0][0], pose_ranges[1][0], pose_ranges[2][0])
        self.pos_hi = wp.vec3f(pose_ranges[0][1], pose_ranges[1][1], pose_ranges[2][1])
        self.rot_lo = wp.vec3f(pose_ranges[3][0], pose_ranges[4][0], pose_ranges[5][0])
        self.rot_hi = wp.vec3f(pose_ranges[3][1], pose_ranges[4][1], pose_ranges[5][1])

        velocity_ranges = [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        self.vel_lin_lo = wp.vec3f(velocity_ranges[0][0], velocity_ranges[1][0], velocity_ranges[2][0])
        self.vel_lin_hi = wp.vec3f(velocity_ranges[0][1], velocity_ranges[1][1], velocity_ranges[2][1])
        self.vel_ang_lo = wp.vec3f(velocity_ranges[3][0], velocity_ranges[4][0], velocity_ranges[5][0])
        self.vel_ang_hi = wp.vec3f(velocity_ranges[3][1], velocity_ranges[4][1], velocity_ranges[5][1])


# ---------------------------------------------------------------------------
# Randomize rigid body center of mass
# ---------------------------------------------------------------------------


@wp.kernel
def _randomize_com_kernel(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    default_body_com_pos_b: wp.array(dtype=wp.vec3f, ndim=2),
    body_com_pos_b: wp.array(dtype=wp.vec3f, ndim=2),
    body_ids: wp.array(dtype=wp.int32),
    com_lo: wp.vec3f,
    com_hi: wp.vec3f,
):
    """Add random offsets to the default center-of-mass positions for selected bodies."""
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    state = rng_state[env_id]
    dx = wp.randf(state, com_lo[0], com_hi[0])
    dy = wp.randf(state, com_lo[1], com_hi[1])
    dz = wp.randf(state, com_lo[2], com_hi[2])
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        v = default_body_com_pos_b[env_id, b]
        body_com_pos_b[env_id, b] = wp.vec3f(v[0] + dx, v[1] + dy, v[2] + dz)
    rng_state[env_id] = state


@WarpCapturable(False, reason="set_coms_mask calls SimulationManager.add_model_change")
def randomize_rigid_body_com(
    env,
    env_mask: wp.array,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the center of mass (CoM) of rigid bodies by adding random offsets.

    Warp-first override of :func:`isaaclab.envs.mdp.events.randomize_rigid_body_com`.
    Writes directly into the sim-bound ``body_com_pos_b`` buffer, then notifies the solver
    via :meth:`set_coms_mask` so it recomputes inertial properties.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _EventStateCache.for_env(env)
    cache_key = (randomize_rigid_body_com, id(asset), id(asset_cfg), _range_cache_key(com_range))
    state = cache.get(cache_key)
    if state is None:
        state = _RandomizeRigidBodyComState(asset, asset_cfg, com_range)
        cache[cache_key] = state

    wp.launch(
        kernel=_randomize_com_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            state.default_com,
            asset.data.body_com_pos_b.warp,
            state.asset_cfg.body_ids_wp,
            state.com_lo,
            state.com_hi,
        ],
        device=env.device,
    )

    # Notify the solver that inertial properties changed (COM position affects inertia).
    asset.set_coms_mask(coms=asset.data.body_com_pos_b.warp, env_mask=env_mask)


# ---------------------------------------------------------------------------
# Apply external force and torque
# ---------------------------------------------------------------------------


@wp.kernel
def _apply_external_force_torque_kernel(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    body_ids: wp.array(dtype=wp.int32),
    force_out: wp.array(dtype=wp.vec3f, ndim=2),
    torque_out: wp.array(dtype=wp.vec3f, ndim=2),
    force_lo: float,
    force_hi: float,
    torque_lo: float,
    torque_hi: float,
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    state = rng_state[env_id]
    for body_index in range(body_ids.shape[0]):
        b = body_ids[body_index]
        force_out[env_id, b] = wp.vec3f(
            wp.randf(state, force_lo, force_hi),
            wp.randf(state, force_lo, force_hi),
            wp.randf(state, force_lo, force_hi),
        )
        torque_out[env_id, b] = wp.vec3f(
            wp.randf(state, torque_lo, torque_hi),
            wp.randf(state, torque_lo, torque_hi),
            wp.randf(state, torque_lo, torque_hi),
        )
    rng_state[env_id] = state


def apply_external_force_torque(
    env,
    env_mask: wp.array,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize external forces and torques applied to the asset's bodies.

    Warp-first override of :func:`isaaclab.envs.mdp.events.apply_external_force_torque`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _EventStateCache.for_env(env)
    cache_key = (
        apply_external_force_torque,
        id(asset),
        id(asset_cfg),
        _range_cache_key(force_range),
        _range_cache_key(torque_range),
    )
    state = cache.get(cache_key)
    if state is None:
        state = _ApplyExternalForceTorqueState(env, asset, asset_cfg, force_range, torque_range)
        cache[cache_key] = state

    wp.launch(
        kernel=_apply_external_force_torque_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            state.body_ids,
            state.forces,
            state.torques,
            state.force_range[0],
            state.force_range[1],
            state.torque_range[0],
            state.torque_range[1],
        ],
        device=env.device,
    )

    asset.permanent_wrench_composer.set_forces_and_torques_mask(
        forces=state.forces,
        torques=state.torques,
        body_mask=state.body_mask,
        env_mask=env_mask,
    )


# ---------------------------------------------------------------------------
# Push by velocity
# ---------------------------------------------------------------------------


@wp.kernel
def _push_by_setting_velocity_kernel(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    vel_out: wp.array(dtype=wp.spatial_vectorf),
    lin_lo: wp.vec3f,
    lin_hi: wp.vec3f,
    ang_lo: wp.vec3f,
    ang_hi: wp.vec3f,
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    vel = root_vel_w[env_id]
    state = rng_state[env_id]

    vel_out[env_id] = wp.spatial_vectorf(
        vel[0] + wp.randf(state, lin_lo[0], lin_hi[0]),
        vel[1] + wp.randf(state, lin_lo[1], lin_hi[1]),
        vel[2] + wp.randf(state, lin_lo[2], lin_hi[2]),
        vel[3] + wp.randf(state, ang_lo[0], ang_hi[0]),
        vel[4] + wp.randf(state, ang_lo[1], ang_hi[1]),
        vel[5] + wp.randf(state, ang_lo[2], ang_hi[2]),
    )

    rng_state[env_id] = state


def push_by_setting_velocity(
    env,
    env_mask: wp.array,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    Warp-first override of :func:`isaaclab.envs.mdp.events.push_by_setting_velocity`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _EventStateCache.for_env(env)
    cache_key = (push_by_setting_velocity, id(asset), id(asset_cfg), _range_cache_key(velocity_range))
    state = cache.get(cache_key)
    if state is None:
        state = _PushBySettingVelocityState(env, asset, asset_cfg, velocity_range)
        cache[cache_key] = state

    wp.launch(
        kernel=_push_by_setting_velocity_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            asset.data.root_vel_w.warp,
            state.velocity,
            state.lin_lo,
            state.lin_hi,
            state.ang_lo,
            state.ang_hi,
        ],
        device=env.device,
    )

    asset.write_root_velocity_to_sim_mask(root_velocity=state.velocity, env_mask=env_mask)


# ---------------------------------------------------------------------------
# Reset root state uniform
# ---------------------------------------------------------------------------


@wp.kernel
def _reset_root_state_uniform_kernel(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    pose_out: wp.array(dtype=wp.transformf),
    vel_out: wp.array(dtype=wp.spatial_vectorf),
    pos_lo: wp.vec3f,
    pos_hi: wp.vec3f,
    rot_lo: wp.vec3f,
    rot_hi: wp.vec3f,
    vel_lin_lo: wp.vec3f,
    vel_lin_hi: wp.vec3f,
    vel_ang_lo: wp.vec3f,
    vel_ang_hi: wp.vec3f,
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    state = rng_state[env_id]

    # --- Pose ---
    default_pose = default_root_pose[env_id]
    default_pos = wp.transform_get_translation(default_pose)
    default_q = wp.transform_get_rotation(default_pose)
    origin = env_origins[env_id]

    # position = default + env_origin + random offset
    pos = wp.vec3f(
        default_pos[0] + origin[0] + wp.randf(state, pos_lo[0], pos_hi[0]),
        default_pos[1] + origin[1] + wp.randf(state, pos_lo[1], pos_hi[1]),
        default_pos[2] + origin[2] + wp.randf(state, pos_lo[2], pos_hi[2]),
    )

    # orientation = default * delta(euler_xyz)
    roll = wp.randf(state, rot_lo[0], rot_hi[0])
    pitch = wp.randf(state, rot_lo[1], rot_hi[1])
    yaw = wp.randf(state, rot_lo[2], rot_hi[2])
    qx = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), roll)
    qy = wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), pitch)
    qz = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), yaw)
    # ZYX extrinsic = XYZ intrinsic: delta = qz * qy * qx
    delta_q = wp.mul(wp.mul(qz, qy), qx)
    final_q = wp.mul(default_q, delta_q)

    pose_out[env_id] = wp.transformf(pos, final_q)

    # --- Velocity ---
    default_vel = default_root_vel[env_id]
    vel_out[env_id] = wp.spatial_vectorf(
        default_vel[0] + wp.randf(state, vel_lin_lo[0], vel_lin_hi[0]),
        default_vel[1] + wp.randf(state, vel_lin_lo[1], vel_lin_hi[1]),
        default_vel[2] + wp.randf(state, vel_lin_lo[2], vel_lin_hi[2]),
        default_vel[3] + wp.randf(state, vel_ang_lo[0], vel_ang_hi[0]),
        default_vel[4] + wp.randf(state, vel_ang_lo[1], vel_ang_hi[1]),
        default_vel[5] + wp.randf(state, vel_ang_lo[2], vel_ang_hi[2]),
    )

    rng_state[env_id] = state


def reset_root_state_uniform(
    env,
    env_mask: wp.array,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    Warp-first override of :func:`isaaclab.envs.mdp.events.reset_root_state_uniform`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _EventStateCache.for_env(env)
    cache_key = (
        reset_root_state_uniform,
        id(asset),
        id(asset_cfg),
        _range_cache_key(pose_range),
        _range_cache_key(velocity_range),
    )
    state = cache.get(cache_key)
    if state is None:
        state = _ResetRootStateUniformState(env, asset, asset_cfg, pose_range, velocity_range)
        cache[cache_key] = state

    wp.launch(
        kernel=_reset_root_state_uniform_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            asset.data.default_root_pose.warp,
            asset.data.default_root_vel.warp,
            env.env_origins_wp,
            state.pose,
            state.velocity,
            state.pos_lo,
            state.pos_hi,
            state.rot_lo,
            state.rot_hi,
            state.vel_lin_lo,
            state.vel_lin_hi,
            state.vel_ang_lo,
            state.vel_ang_hi,
        ],
        device=env.device,
    )

    asset.write_root_pose_to_sim_mask(root_pose=state.pose, env_mask=env_mask)
    asset.write_root_velocity_to_sim_mask(root_velocity=state.velocity, env_mask=env_mask)


# ---------------------------------------------------------------------------
# Reset joints by offset
# ---------------------------------------------------------------------------


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
            asset.data.default_joint_pos.warp,
            asset.data.default_joint_vel.warp,
            asset.data.joint_pos.warp,
            asset.data.joint_vel.warp,
            asset.data.soft_joint_pos_limits.warp,
            asset.data.soft_joint_vel_limits.warp,
            float(position_range[0]),
            float(position_range[1]),
            float(velocity_range[0]),
            float(velocity_range[1]),
        ],
        device=env.device,
    )

    # Sync derived buffers (_previous_joint_vel, joint_acc) for reset envs.
    asset.write_joint_position_to_sim_mask(position=asset.data.joint_pos.warp, env_mask=env_mask)
    asset.write_joint_velocity_to_sim_mask(velocity=asset.data.joint_vel.warp, env_mask=env_mask)


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
            asset.data.default_joint_pos.warp,
            asset.data.default_joint_vel.warp,
            asset.data.joint_pos.warp,
            asset.data.joint_vel.warp,
            asset.data.soft_joint_pos_limits.warp,
            asset.data.soft_joint_vel_limits.warp,
            float(position_range[0]),
            float(position_range[1]),
            float(velocity_range[0]),
            float(velocity_range[1]),
        ],
        device=env.device,
    )

    # Sync derived buffers (_previous_joint_vel, joint_acc) for reset envs.
    asset.write_joint_position_to_sim_mask(position=asset.data.joint_pos.warp, env_mask=env_mask)
    asset.write_joint_velocity_to_sim_mask(velocity=asset.data.joint_vel.warp, env_mask=env_mask)
