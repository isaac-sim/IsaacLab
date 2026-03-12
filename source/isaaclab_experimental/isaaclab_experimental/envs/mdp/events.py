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

import warp as wp

from isaaclab.assets import Articulation

from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.utils.warp import WarpCapturable

# ---------------------------------------------------------------------------
# Randomize rigid body center of mass
# ---------------------------------------------------------------------------


@wp.kernel
def _randomize_com_kernel(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    body_com_pos_b: wp.array(dtype=wp.vec3f, ndim=2),
    body_ids: wp.array(dtype=wp.int32),
    com_lo: wp.vec3f,
    com_hi: wp.vec3f,
):
    """Add random offset to center of mass positions for selected bodies."""
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    state = rng_state[env_id]
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        v = body_com_pos_b[env_id, b]
        dx = wp.randf(state, com_lo[0], com_hi[0])
        dy = wp.randf(state, com_lo[1], com_hi[1])
        dz = wp.randf(state, com_lo[2], com_hi[2])
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

    fn = randomize_rigid_body_com
    if not hasattr(fn, "_com_lo") or fn._asset_name != asset_cfg.name:
        fn._asset_name = asset_cfg.name
        r = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        fn._com_lo = wp.vec3f(r[0][0], r[1][0], r[2][0])
        fn._com_hi = wp.vec3f(r[0][1], r[1][1], r[2][1])

    wp.launch(
        kernel=_randomize_com_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            asset.data.body_com_pos_b,
            asset_cfg.body_ids_wp,
            fn._com_lo,
            fn._com_hi,
        ],
        device=env.device,
    )

    # Notify the solver that inertial properties changed (COM position affects inertia).
    asset.set_coms_mask(coms=asset.data.body_com_pos_b, env_mask=env_mask)


# ---------------------------------------------------------------------------
# Apply external force and torque
# ---------------------------------------------------------------------------


@wp.kernel
def _apply_external_force_torque_kernel(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    force_out: wp.array(dtype=wp.vec3f, ndim=2),
    torque_out: wp.array(dtype=wp.vec3f, ndim=2),
    force_lo: float,
    force_hi: float,
    torque_lo: float,
    torque_hi: float,
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        # zero out unmasked envs so they don't accumulate stale forces
        for b in range(force_out.shape[1]):
            force_out[env_id, b] = wp.vec3f(0.0, 0.0, 0.0)
            torque_out[env_id, b] = wp.vec3f(0.0, 0.0, 0.0)
        return

    state = rng_state[env_id]
    for b in range(force_out.shape[1]):
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

    # First-call: allocate scratch and pre-convert constant arguments.
    if not hasattr(apply_external_force_torque, "_scratch_forces"):
        apply_external_force_torque._scratch_forces = wp.zeros(
            (env.num_envs, asset.num_bodies), dtype=wp.vec3f, device=env.device
        )
        apply_external_force_torque._scratch_torques = wp.zeros(
            (env.num_envs, asset.num_bodies), dtype=wp.vec3f, device=env.device
        )

    wp.launch(
        kernel=_apply_external_force_torque_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            apply_external_force_torque._scratch_forces,
            apply_external_force_torque._scratch_torques,
            force_range[0],
            force_range[1],
            torque_range[0],
            torque_range[1],
        ],
        device=env.device,
    )

    asset.permanent_wrench_composer.set_forces_and_torques_mask(
        forces=apply_external_force_torque._scratch_forces,
        torques=apply_external_force_torque._scratch_torques,
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

    # First-call: allocate scratch and pre-parse constant range arguments.
    if not hasattr(push_by_setting_velocity, "_scratch_vel"):
        push_by_setting_velocity._scratch_vel = wp.zeros((env.num_envs,), dtype=wp.spatial_vectorf, device=env.device)
        r = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        push_by_setting_velocity._lin_lo = wp.vec3f(r[0][0], r[1][0], r[2][0])
        push_by_setting_velocity._lin_hi = wp.vec3f(r[0][1], r[1][1], r[2][1])
        push_by_setting_velocity._ang_lo = wp.vec3f(r[3][0], r[4][0], r[5][0])
        push_by_setting_velocity._ang_hi = wp.vec3f(r[3][1], r[4][1], r[5][1])

    wp.launch(
        kernel=_push_by_setting_velocity_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            asset.data.root_vel_w,
            push_by_setting_velocity._scratch_vel,
            push_by_setting_velocity._lin_lo,
            push_by_setting_velocity._lin_hi,
            push_by_setting_velocity._ang_lo,
            push_by_setting_velocity._ang_hi,
        ],
        device=env.device,
    )

    asset.write_root_velocity_to_sim_mask(root_velocity=push_by_setting_velocity._scratch_vel, env_mask=env_mask)


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

    # First-call: allocate scratch and pre-parse range dicts.
    if not hasattr(reset_root_state_uniform, "_scratch_pose"):
        reset_root_state_uniform._scratch_pose = wp.zeros((env.num_envs,), dtype=wp.transformf, device=env.device)
        reset_root_state_uniform._scratch_vel = wp.zeros((env.num_envs,), dtype=wp.spatial_vectorf, device=env.device)
        # Pre-parse pose_range dict
        p = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        reset_root_state_uniform._pos_lo = wp.vec3f(p[0][0], p[1][0], p[2][0])
        reset_root_state_uniform._pos_hi = wp.vec3f(p[0][1], p[1][1], p[2][1])
        reset_root_state_uniform._rot_lo = wp.vec3f(p[3][0], p[4][0], p[5][0])
        reset_root_state_uniform._rot_hi = wp.vec3f(p[3][1], p[4][1], p[5][1])
        # Pre-parse velocity_range dict
        v = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        reset_root_state_uniform._vel_lin_lo = wp.vec3f(v[0][0], v[1][0], v[2][0])
        reset_root_state_uniform._vel_lin_hi = wp.vec3f(v[0][1], v[1][1], v[2][1])
        reset_root_state_uniform._vel_ang_lo = wp.vec3f(v[3][0], v[4][0], v[5][0])
        reset_root_state_uniform._vel_ang_hi = wp.vec3f(v[3][1], v[4][1], v[5][1])

    wp.launch(
        kernel=_reset_root_state_uniform_kernel,
        dim=env.num_envs,
        inputs=[
            env_mask,
            env.rng_state_wp,
            asset.data.default_root_pose,
            asset.data.default_root_vel,
            env.env_origins_wp,
            reset_root_state_uniform._scratch_pose,
            reset_root_state_uniform._scratch_vel,
            reset_root_state_uniform._pos_lo,
            reset_root_state_uniform._pos_hi,
            reset_root_state_uniform._rot_lo,
            reset_root_state_uniform._rot_hi,
            reset_root_state_uniform._vel_lin_lo,
            reset_root_state_uniform._vel_lin_hi,
            reset_root_state_uniform._vel_ang_lo,
            reset_root_state_uniform._vel_ang_hi,
        ],
        device=env.device,
    )

    asset.write_root_pose_to_sim_mask(root_pose=reset_root_state_uniform._scratch_pose, env_mask=env_mask)
    asset.write_root_velocity_to_sim_mask(root_velocity=reset_root_state_uniform._scratch_vel, env_mask=env_mask)


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

    # Sync derived buffers (_previous_joint_vel, joint_acc) for reset envs.
    asset.write_joint_position_to_sim_mask(position=asset.data.joint_pos, env_mask=env_mask)
    asset.write_joint_velocity_to_sim_mask(velocity=asset.data.joint_vel, env_mask=env_mask)


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

    # Sync derived buffers (_previous_joint_vel, joint_acc) for reset envs.
    asset.write_joint_position_to_sim_mask(position=asset.data.joint_pos, env_mask=env_mask)
    asset.write_joint_velocity_to_sim_mask(velocity=asset.data.joint_vel, env_mask=env_mask)
