# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first overrides for common event terms.

These terms are intended to be used with the experimental Warp-first
:class:`isaaclab_experimental.managers.EventManager` (mask-based interval/reset).

Why this exists:
- Stable event terms (e.g. `isaaclab.envs.mdp.events.reset_joints_by_offset`) often build torch tensors and then
  call into Newton articulation writers with partial indices (env_ids/joint_ids).
- On the Newton backend, passing torch tensors triggers expensive torch->warp conversions that currently allocate
  full `(num_envs, num_joints)` buffers.

These Warp-first implementations avoid that by writing directly into the sim-bound Warp state buffers
(`asset.data.joint_pos` / `asset.data.joint_vel`) for the selected envs/joints.

Stateful terms use :class:`~isaaclab_experimental.managers.ManagerTermBase` so
persistent buffers and parsed constants are created once during manager setup.

Notes:
- These terms assume the Newton/Warp backend (Warp arrays are available for joint state and defaults).
- For best performance, pass :class:`isaaclab_experimental.managers.SceneEntityCfg` so `joint_ids_wp` is cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.events import randomize_rigid_body_mass as _StableRandomizeRigidBodyMass
from isaaclab.envs.mdp.events import randomize_rigid_body_material as _StableRandomizeRigidBodyMaterial

from isaaclab_experimental.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_experimental.managers import ManagerTermBase as _WarpManagerTermBase
from isaaclab_experimental.utils.warp import WarpCapturable

__all__ = [
    "apply_external_force_torque",
    "push_by_setting_velocity",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_material",
    "reset_joints_by_offset",
    "reset_joints_by_scale",
    "reset_root_state_uniform",
]

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _resolve_body_ids(asset_cfg: SceneEntityCfg, num_bodies: int) -> list[int]:
    """Resolve a configured body selection without allocating device storage."""
    if asset_cfg.body_ids is None or asset_cfg.body_ids == slice(None):
        return list(range(num_bodies))
    if isinstance(asset_cfg.body_ids, int):
        return [asset_cfg.body_ids]
    return list(asset_cfg.body_ids)


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
class randomize_rigid_body_com(ManagerTermBase):
    """Randomize rigid-body centers of mass from a persistent default baseline.

    This term is not CUDA-graph capturable because notifying the solver of changed
    inertial properties calls :meth:`SimulationManager.add_model_change`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize persistent center-of-mass randomization state.

        Args:
            cfg: Event term configuration.
            env: Environment containing the randomized asset.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        self._default_com = wp.clone(self._asset.data.body_com_pos_b.warp)
        body_ids = _resolve_body_ids(asset_cfg, self._asset.num_bodies)
        self._body_ids = wp.array(body_ids, dtype=wp.int32, device=env.device)

        com_range = cfg.params["com_range"]
        ranges = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
        self._com_lo = wp.vec3f(ranges[0][0], ranges[1][0], ranges[2][0])
        self._com_hi = wp.vec3f(ranges[0][1], ranges[1][1], ranges[2][1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_mask: wp.array(dtype=wp.bool),
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Randomize selected center-of-mass offsets [m].

        Args:
            env: Environment containing the randomized asset.
            env_mask: Boolean Warp mask selecting environments.
            com_range: Per-axis offset ranges [m]. Parsed during initialization.
            asset_cfg: Scene entity selection. Resolved during initialization.
        """
        wp.launch(
            kernel=_randomize_com_kernel,
            dim=env.num_envs,
            inputs=[
                env_mask,
                env.rng_state_wp,
                self._default_com,
                self._asset.data.body_com_pos_b.warp,
                self._body_ids,
                self._com_lo,
                self._com_hi,
            ],
            device=env.device,
        )

        self._asset.set_coms_mask(coms=self._asset.data.body_com_pos_b.warp, env_mask=env_mask)


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
    """Sample uniform random external force [N] and torque [N·m] vectors for

    selected envs' target bodies into the asset's wrench composer buffers.
    """
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


class apply_external_force_torque(ManagerTermBase):
    """Apply random external forces and torques using persistent wrench buffers."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize persistent external-wrench state.

        Args:
            cfg: Event term configuration.
            env: Environment containing the randomized asset.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        self._forces = wp.zeros((env.num_envs, self._asset.num_bodies), dtype=wp.vec3f, device=env.device)
        self._torques = wp.zeros((env.num_envs, self._asset.num_bodies), dtype=wp.vec3f, device=env.device)

        body_ids = _resolve_body_ids(asset_cfg, self._asset.num_bodies)
        body_mask = [False] * self._asset.num_bodies
        for body_id in body_ids:
            body_mask[body_id] = True
        self._body_ids = wp.array(body_ids, dtype=wp.int32, device=env.device)
        self._body_mask = wp.array(body_mask, dtype=wp.bool, device=env.device)

        force_range = cfg.params["force_range"]
        torque_range = cfg.params["torque_range"]
        self._force_lo = float(force_range[0])
        self._force_hi = float(force_range[1])
        self._torque_lo = float(torque_range[0])
        self._torque_hi = float(torque_range[1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_mask: wp.array(dtype=wp.bool),
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Apply sampled forces [N] and torques [N·m] to selected bodies.

        Args:
            env: Environment containing the randomized asset.
            env_mask: Boolean Warp mask selecting environments.
            force_range: Component-wise force range [N]. Parsed during initialization.
            torque_range: Component-wise torque range [N·m]. Parsed during initialization.
            asset_cfg: Scene entity selection. Resolved during initialization.
        """
        wp.launch(
            kernel=_apply_external_force_torque_kernel,
            dim=env.num_envs,
            inputs=[
                env_mask,
                env.rng_state_wp,
                self._body_ids,
                self._forces,
                self._torques,
                self._force_lo,
                self._force_hi,
                self._torque_lo,
                self._torque_hi,
            ],
            device=env.device,
        )

        self._asset.permanent_wrench_composer.set_forces_and_torques_mask(
            forces=self._forces,
            torques=self._torques,
            body_mask=self._body_mask,
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
    """Add a uniform random velocity kick [m/s, rad/s] to selected envs' current

    root velocity, writing the result for the masked sim write.
    """
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


class push_by_setting_velocity(ManagerTermBase):
    """Push an asset by sampling into a persistent root-velocity buffer."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize persistent root-velocity push state.

        Args:
            cfg: Event term configuration.
            env: Environment containing the pushed asset.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        self._velocity = wp.zeros(env.num_envs, dtype=wp.spatial_vectorf, device=env.device)

        velocity_range = cfg.params["velocity_range"]
        ranges = [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        self._lin_lo = wp.vec3f(ranges[0][0], ranges[1][0], ranges[2][0])
        self._lin_hi = wp.vec3f(ranges[0][1], ranges[1][1], ranges[2][1])
        self._ang_lo = wp.vec3f(ranges[3][0], ranges[4][0], ranges[5][0])
        self._ang_hi = wp.vec3f(ranges[3][1], ranges[4][1], ranges[5][1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_mask: wp.array(dtype=wp.bool),
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Add sampled linear [m/s] and angular [rad/s] root velocities.

        Args:
            env: Environment containing the pushed asset.
            env_mask: Boolean Warp mask selecting environments.
            velocity_range: Per-axis velocity ranges [m/s or rad/s]. Parsed during initialization.
            asset_cfg: Scene entity selection. Resolved during initialization.
        """
        wp.launch(
            kernel=_push_by_setting_velocity_kernel,
            dim=env.num_envs,
            inputs=[
                env_mask,
                env.rng_state_wp,
                self._asset.data.root_vel_w.warp,
                self._velocity,
                self._lin_lo,
                self._lin_hi,
                self._ang_lo,
                self._ang_hi,
            ],
            device=env.device,
        )

        self._asset.write_root_velocity_to_sim_mask(root_velocity=self._velocity, env_mask=env_mask)


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
    """Compose selected envs' reset root state: default pose offset by the env

    origin plus uniform position [m] / roll-pitch-yaw [rad] noise, and default
    velocity plus uniform noise [m/s, rad/s].
    """
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


class reset_root_state_uniform(ManagerTermBase):
    """Reset root pose and velocity using persistent Warp output buffers."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize persistent root-state reset state.

        Args:
            cfg: Event term configuration.
            env: Environment containing the reset asset.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        self._default_root_pose = self._asset.data.default_root_pose.warp
        self._default_root_velocity = self._asset.data.default_root_vel.warp
        self._env_origins = env.env_origins_wp
        self._pose = wp.zeros(env.num_envs, dtype=wp.transformf, device=env.device)
        self._velocity = wp.zeros(env.num_envs, dtype=wp.spatial_vectorf, device=env.device)

        pose_range = cfg.params["pose_range"]
        pose_ranges = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        self._pos_lo = wp.vec3f(pose_ranges[0][0], pose_ranges[1][0], pose_ranges[2][0])
        self._pos_hi = wp.vec3f(pose_ranges[0][1], pose_ranges[1][1], pose_ranges[2][1])
        self._rot_lo = wp.vec3f(pose_ranges[3][0], pose_ranges[4][0], pose_ranges[5][0])
        self._rot_hi = wp.vec3f(pose_ranges[3][1], pose_ranges[4][1], pose_ranges[5][1])

        velocity_range = cfg.params["velocity_range"]
        velocity_ranges = [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        self._vel_lin_lo = wp.vec3f(velocity_ranges[0][0], velocity_ranges[1][0], velocity_ranges[2][0])
        self._vel_lin_hi = wp.vec3f(velocity_ranges[0][1], velocity_ranges[1][1], velocity_ranges[2][1])
        self._vel_ang_lo = wp.vec3f(velocity_ranges[3][0], velocity_ranges[4][0], velocity_ranges[5][0])
        self._vel_ang_hi = wp.vec3f(velocity_ranges[3][1], velocity_ranges[4][1], velocity_ranges[5][1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_mask: wp.array(dtype=wp.bool),
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Reset root pose [m, rad] and velocity [m/s, rad/s].

        Args:
            env: Environment containing the reset asset.
            env_mask: Boolean Warp mask selecting environments.
            pose_range: Position and Euler-angle ranges [m or rad]. Parsed during initialization.
            velocity_range: Linear and angular velocity ranges [m/s or rad/s]. Parsed during initialization.
            asset_cfg: Scene entity selection. Resolved during initialization.
        """
        wp.launch(
            kernel=_reset_root_state_uniform_kernel,
            dim=env.num_envs,
            inputs=[
                env_mask,
                env.rng_state_wp,
                self._default_root_pose,
                self._default_root_velocity,
                self._env_origins,
                self._pose,
                self._velocity,
                self._pos_lo,
                self._pos_hi,
                self._rot_lo,
                self._rot_hi,
                self._vel_lin_lo,
                self._vel_lin_hi,
                self._vel_ang_lo,
                self._vel_ang_hi,
            ],
            device=env.device,
        )

        self._asset.write_root_pose_to_sim_mask(root_pose=self._pose, env_mask=env_mask)
        self._asset.write_root_velocity_to_sim_mask(root_velocity=self._velocity, env_mask=env_mask)


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
    """Reset selected envs' selected joints to defaults plus uniform offsets

    [rad or m, depending on joint type], clamped to the soft position and
    velocity limits.
    """
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
    env: ManagerBasedEnv,
    env_mask: wp.array(dtype=wp.bool),
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
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
    """Reset selected envs' selected joints to defaults scaled by uniform random

    factors, clamped to the soft position and velocity limits.
    """
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
    env: ManagerBasedEnv,
    env_mask: wp.array(dtype=wp.bool),
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
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


# ---------------------------------------------------------------------------
# Startup-mode randomization (mask-signature adapters)
# ---------------------------------------------------------------------------
#
# The stable material/mass randomization terms already dispatch to the active
# physics backend (Newton included); only their calling convention differs —
# the warp EventManager invokes terms with a Warp env-mask while the stable
# class terms expect torch env indices. Both terms run in ``startup`` mode,
# once at construction and outside any captured path, so a host-side
# mask-to-ids conversion is acceptable.


def _mask_to_env_ids(env_mask: wp.array) -> torch.Tensor:
    """Convert a Warp boolean env-mask to the torch index tensor the stable terms expect."""
    return torch.nonzero(wp.to_torch(env_mask), as_tuple=False).squeeze(-1)


class randomize_rigid_body_material(_StableRandomizeRigidBodyMaterial, _WarpManagerTermBase):
    """Warp adapter for the stable, backend-dispatched material randomization term.

    Converts the warp event manager's env-mask calling convention to the stable
    term's env-ids convention and delegates. Startup mode only. Inherits the
    warp :class:`ManagerTermBase` so the warp managers accept it as a class term.
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_mask: wp.array,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ) -> None:
        super().__call__(
            env,
            _mask_to_env_ids(env_mask),
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent,
        )

    def reset(self, env_mask: wp.array | None = None) -> None:
        """Nothing to reset; materials are randomized once at startup."""
        return


class randomize_rigid_body_mass(_StableRandomizeRigidBodyMass, _WarpManagerTermBase):
    """Warp adapter for the stable, backend-dispatched mass randomization term.

    Converts the warp event manager's env-mask calling convention to the stable
    term's env-ids convention and delegates. Startup mode only. Inherits the
    warp :class:`ManagerTermBase` so the warp managers accept it as a class term.
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_mask: wp.array,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: str,
        distribution: str = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
    ) -> None:
        super().__call__(
            env,
            _mask_to_env_ids(env_mask),
            asset_cfg,
            mass_distribution_params,
            operation,
            distribution,
            recompute_inertia,
            min_mass,
        )

    def reset(self, env_mask: wp.array | None = None) -> None:
        """Nothing to reset; masses are randomized once at startup."""
        return
