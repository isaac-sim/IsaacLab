# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking, posture, recovery, skating and manipulation reward terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import body_projected_gravity_b
from isaaclab.envs.mdp import joint_pos_limits as _joint_pos_limits
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.string import resolve_matching_names_values

from .observations import foot_height, joint_pos_rel_biased, zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

import math
from collections.abc import Mapping

from . import observations as _observations
from .events import ball_kick_direction, roulade_roll_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _command_magnitude(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """scalar command magnitude ``|v_xy| + |w_z|``."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


"""
Velocity-tracking rewards.
"""


def track_linear_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track linear velocity, including vertical drift, with tolerance ``std`` [m/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b.torch
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return zero_non_finite(torch.exp(-(xy_error + z_error) / std**2))


def track_angular_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track yaw rate, including roll/pitch drift, with tolerance ``std`` [rad/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b.torch
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return zero_non_finite(torch.exp(-(z_error + xy_error) / std**2))


def track_linear_velocity_heading(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track horizontal root-link velocity in the heading frame with tolerance ``std`` [m/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.torch), asset.data.root_link_lin_vel_w.torch
    )
    error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    return zero_non_finite(torch.exp(-error / std**2))


def track_angular_velocity_world(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track world-frame yaw rate with tolerance ``std`` [rad/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w.torch[:, 2])
    return zero_non_finite(torch.exp(-error / std**2))


"""
Posture rewards.
"""


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward keeping one body's frame aligned with gravity, using a Gaussian kernel."""
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    xy_squared = torch.nan_to_num(xy_squared, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.exp(-xy_squared / std**2)


class joint_deviation_exp(ManagerTermBase):
    """Reward staying near the default joint positions, with a per-joint tolerance that widens once the robot moves.

    The tolerance is ``std_standing`` [rad] while the command magnitude is below ``walking_threshold``, and
    ``std_walking`` [rad] otherwise. With ``standing_only=True``, walking commands receive zero reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve the standing and walking per-joint standard deviations."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        joint_names = [env.scene[asset_cfg.name].joint_names[index] for index in asset_cfg.joint_ids]
        self._std = {}
        for key in ("std_standing", "std_walking"):
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            self._std[key] = torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        walking_threshold: float,
        asset_cfg: SceneEntityCfg,
        standing_only: bool = False,
    ) -> torch.Tensor:
        """Compute the posture reward."""
        asset: Articulation = env.scene[asset_cfg.name]
        standing = (_command_magnitude(env, command_name) < walking_threshold).unsqueeze(1)
        std = torch.where(standing, self._std["std_standing"], self._std["std_walking"])
        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        error_squared = torch.square(joint_pos - default_joint_pos)
        reward = zero_non_finite(torch.exp(-torch.mean(error_squared / std**2, dim=1)))
        return reward * standing.squeeze(1) if standing_only else reward


"""
Head-pose rewards.
"""


def head_pose_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    backlash_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Track head encoder offsets with tolerance ``std`` [rad], including play when configured."""
    command = env.command_manager.get_command(command_name)
    error = joint_pos_rel_biased(env, asset_cfg, biased=False, backlash_cfg=backlash_cfg) - command
    return zero_non_finite(torch.exp(-((error / std) ** 2)).mean(dim=-1))


class head_pose_bias_penalty(ManagerTermBase):
    """Penalize a head-pose error that persists, as opposed to one that averages out."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment moving average."""
        super().__init__(cfg, env)
        self._error_ema = torch.zeros(env.num_envs, len(cfg.params["asset_cfg"].joint_ids), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the average for the environments that restarted."""
        if env_ids is None:
            self._error_ema[:] = 0.0
        else:
            self._error_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tau_s: float,
        asset_cfg: SceneEntityCfg,
        backlash_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Return the absolute error averaged with time constant ``tau_s`` [s]."""
        command = env.command_manager.get_command(command_name)
        error = zero_non_finite(joint_pos_rel_biased(env, asset_cfg, biased=False, backlash_cfg=backlash_cfg) - command)
        alpha = min(1.0, env.step_dt / tau_s)
        self._error_ema = (1.0 - alpha) * self._error_ema + alpha * error
        return zero_non_finite(self._error_ema.abs().mean(dim=-1))


"""
Gait and foot rewards.
"""


def feet_air_time_windowed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward air time strictly between ``threshold_min`` and ``threshold_max`` [s] while moving."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    reward = reward * (_command_magnitude(env, command_name) > command_threshold).float()
    return reward


def foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    height_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Penalize moving feet that deviate from ``target_height`` [m] above the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    heights = foot_height(env, asset_cfg, height_sensor_names)
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.abs(heights - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1)
    return zero_non_finite(cost * (_command_magnitude(env, command_name) > command_threshold).float())


class foot_swing_height(ManagerTermBase):
    """Penalize, on touchdown, how far the peak of the swing was from a target height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-foot peak heights."""
        super().__init__(cfg, env)
        num_feet = len(cfg.params["sensor_cfg"].body_ids)
        self._peak_heights = torch.zeros(env.num_envs, num_feet, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the tracked peaks for the environments that restarted."""
        if env_ids is None:
            self._peak_heights[:] = 0.0
        else:
            self._peak_heights[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
        height_sensor_names: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Penalize touchdown error relative to ``target_height`` [m]."""
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_air = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        heights = zero_non_finite(foot_height(env, asset_cfg, height_sensor_names))
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, heights), self._peak_heights)
        first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        cost = cost * (_command_magnitude(env, command_name) > command_threshold).float()
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return zero_non_finite(cost)


def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the squared planar speed of feet that are on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = (contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
    return zero_non_finite(cost * (_command_magnitude(env, command_name) > command_threshold).float())


"""
Regularization penalties.
"""


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the roll and pitch rate of selected bodies, in the world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return zero_non_finite(torch.sum(torch.square(ang_vel_xy), dim=(1, 2)))


def angular_momentum_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the squared magnitude of the articulation's angular momentum about its own centre of mass."""
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.data.body_mass.torch[:, asset_cfg.body_ids]
    pos_w = asset.data.body_com_pos_w.torch[:, asset_cfg.body_ids]
    lin_vel_w = asset.data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_com_ang_vel_w.torch[:, asset_cfg.body_ids]
    quat_w = asset.data.body_com_quat_w.torch[:, asset_cfg.body_ids]
    inertia_b = asset.data.body_inertia.torch[:, asset_cfg.body_ids].view(*mass.shape, 3, 3)
    total_mass = mass.sum(dim=1, keepdim=True).unsqueeze(-1)
    com_pos_w = (mass.unsqueeze(-1) * pos_w).sum(dim=1, keepdim=True) / total_mass
    com_lin_vel_w = (mass.unsqueeze(-1) * lin_vel_w).sum(dim=1, keepdim=True) / total_mass
    rot = math_utils.matrix_from_quat(quat_w)
    inertia_w = torch.matmul(rot, torch.matmul(inertia_b, rot.transpose(-2, -1)))
    spin = torch.matmul(inertia_w, ang_vel_w.unsqueeze(-1)).squeeze(-1)
    orbit = mass.unsqueeze(-1) * torch.cross(pos_w - com_pos_w, lin_vel_w - com_lin_vel_w, dim=-1)
    momentum = (spin + orbit).sum(dim=1)
    return zero_non_finite(torch.sum(torch.square(momentum), dim=-1))


def joint_pos_limits_finite(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """:func:`~isaaclab.envs.mdp.joint_pos_limits`, zero for environments whose state is not finite."""
    return zero_non_finite(_joint_pos_limits(env, asset_cfg))


"""
Contact penalties.
"""


def self_collision_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize parts of the robot that are touching the robot itself."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(f"{sensor_cfg.name} needs a filtered contact sensor to measure self-contacts.")
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
    touching = in_contact.any(dim=-1)
    return touching.sum(dim=-1).float()


def _required_entity_cfg(cfg: RewardTermCfg, key: str, term_name: str) -> SceneEntityCfg:
    """Fetch a scene entity configuration that a stateful term needs at construction time.

    A class-based term is built before it is ever called, so it cannot fall back on a signature
    default: the entity has to be spelled out in :attr:`RewardTermCfg.params`. Reading it directly
    would surface as a bare ``KeyError`` from inside the manager, which says nothing about what to
    fix.

    Args:
        cfg: The term configuration.
        key: Name of the parameter holding the entity configuration.
        term_name: Name of the term, used in the error message.

    Returns:
        The scene entity configuration.

    Raises:
        ValueError: If the parameter is missing or is not a :class:`SceneEntityCfg`.
    """
    entity_cfg = cfg.params.get(key)
    if not isinstance(entity_cfg, SceneEntityCfg):
        raise ValueError(
            f"The reward term '{term_name}' requires a 'SceneEntityCfg' under the '{key}' entry of"
            f" its 'params'. Received: {entity_cfg!r}."
        )
    return entity_cfg


def _required_joint_ids(entity_cfg: SceneEntityCfg, key: str, term_name: str) -> list[int]:
    """Fetch an explicit joint selection from a resolved scene entity configuration.

    An unset :attr:`SceneEntityCfg.joint_names` leaves :attr:`~SceneEntityCfg.joint_ids` as
    ``slice(None)``, which selects every joint in whatever order the backend resolved them. The
    terms that use this helper index a command tensor positionally, so silently selecting all
    joints would mis-pair the columns rather than fail.

    Args:
        entity_cfg: The resolved scene entity configuration.
        key: Name of the parameter holding it, used in the error message.
        term_name: Name of the term, used in the error message.

    Returns:
        The selected joint indices.

    Raises:
        ValueError: If no joints were selected by name.
    """
    if isinstance(entity_cfg.joint_ids, slice):
        raise ValueError(
            f"The reward term '{term_name}' requires '{key}' to select its joints by name, so that"
            " their order is pinned. Set 'joint_names' with 'preserve_order=True'."
        )
    return entity_cfg.joint_ids


def _root_height_above_ground(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    """Height of the articulation's root link above the ground its environment sits on [m]."""
    return torch.nan_to_num(asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _trunk_tilt_squared(quat: torch.Tensor) -> torch.Tensor:
    """``2 * (q_x^2 + q_y^2)`` tilt measure, which is ``1 - cos(tilt)``.

    Zero when the body frame is upright, 1 when it is horizontal and 2 when it is inverted.

    Args:
        quat: Root link orientation in (x, y, z, w). Shape is (num_envs, 4).

    Returns:
        The squared tilt in ``[0, 2]``. Shape is (num_envs,).
    """
    return 2.0 * (torch.square(quat[:, 0]) + torch.square(quat[:, 1]))


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    """``t^2 (3 - 2t)`` smoothstep on an already-normalized, unclamped ratio."""
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _height_gate(z: torch.Tensor, height_low: float, height_high: float) -> torch.Tensor:
    """Smoothstep gate that opens as the trunk rises from ``height_low`` [m] to ``height_high`` [m]."""
    return _smoothstep((z - height_low) / max(height_high - height_low, 1e-6))


def _tilt_gate(quat: torch.Tensor, tilt_full_deg: float, tilt_zero_deg: float) -> torch.Tensor:
    """Smoothstep gate that closes as the trunk tilts from ``tilt_full_deg`` to ``tilt_zero_deg``."""
    cos_tilt = 1.0 - _trunk_tilt_squared(quat)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
    return _smoothstep((tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6))


def _cos_tilt_gate(quat: torch.Tensor, tilt_full_deg: float, tilt_zero_deg: float) -> torch.Tensor:
    """The same gate interpolated in ``cos(tilt)`` rather than in the angle."""
    cos_tilt = 1.0 - _trunk_tilt_squared(quat)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    return _smoothstep((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6))


def _is_fallen(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    tilt_above_deg: float,
    z_below: float | None = None,
) -> torch.Tensor:
    """Whether the trunk counts as fallen: tilted past ``tilt_above_deg``, or below ``z_below`` [m].

    Args:
        env: The environment instance.
        asset: The articulation whose root link carries the trunk.
        tilt_above_deg: Trunk tilt [deg] beyond which the robot counts as fallen.
        z_below: Trunk height [m] below which it counts as fallen regardless of tilt. Defaults to
            None, which tests the tilt alone.

    Returns:
        Whether each environment's trunk is fallen. Shape is (num_envs,).
    """
    cos_tilt = 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)
    fallen = cos_tilt < math.cos(math.radians(tilt_above_deg))
    if z_below is not None:
        fallen |= _root_height_above_ground(env, asset) < z_below
    return fallen


def _feet_height_above_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of the selected foot bodies above the ground [m], shape (num_envs, num_feet)."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_z = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    return foot_pos_z - env.scene.env_origins[:, 2].unsqueeze(1)


"""
Velocity tracking.
"""


def track_linear_velocity_raw(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the linear velocity command, penalizing vertical drift in the same kernel.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel [m/s].
        command_name: Name of the velocity command term.
        asset_cfg: The articulation to measure. Defaults to the entity named ``"robot"``.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b.torch
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    # the commanded vertical velocity is always zero, so the drift itself is the error
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity_raw(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the yaw-rate command, penalizing roll and pitch rate in the same kernel.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel [rad/s].
        command_name: Name of the velocity command term.
        asset_cfg: The articulation to measure. Defaults to the entity named ``"robot"``.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b.torch
    z_error = torch.square(command[:, 2] - actual[:, 2])
    # the commanded roll and pitch rates are always zero
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)


"""
Posture.
"""


def upright_selected_body(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward keeping a body's frame aligned with gravity, using a Gaussian kernel.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel, in units of the projected gravity direction.
        asset_cfg: The articulation and the single body to measure. With no body selected the
            articulation root is used, as does.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).

    Note:
        The tilt is NaN-guarded to its own worst case, so an environment whose frame has stopped
        being a number earns nothing rather than poisoning the batch or collecting full marks. See
        :func:`joint_pose_l2` for why no termination can catch this before the reward is computed.

    Raises:
        ValueError: If ``asset_cfg`` selects more than one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None:
        projected_gravity_b = asset.data.projected_gravity_b.torch
    else:
        if len(asset_cfg.body_ids) != 1:
            raise ValueError(
                f"'upright' measures a single body's tilt; 'asset_cfg' selected"
                f" {len(asset_cfg.body_ids)} bodies: {asset_cfg.body_names}."
            )
        body_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
        gravity_dir_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1).unsqueeze(1)
        projected_gravity_b = math_utils.quat_apply_inverse(body_quat_w, gravity_dir_w).squeeze(1)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    # A diverged solver can leave the measured frame non-finite for one step, and the reset that
    # repairs it runs after the reward is computed, so no termination can keep the NaN out of the
    # reward buffer RSL-RL checks. The quantity is a squared sine and therefore lives in [0, 1], so
    # an unmeasurable frame is scored as *fully* tilted -- the conservative end of its own range,
    # which pays this environment nothing rather than the full marks a zero would. No-op on any
    # finite frame.
    xy_squared = torch.nan_to_num(xy_squared, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.exp(-xy_squared / std**2)


class pose_mode_switch(ManagerTermBase):
    """Reward holding the stand pose, with a per-joint tolerance that widens once the robot moves."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve the three per-joint standard-deviation vectors.

        Args:
            cfg: The term configuration, whose ``params`` carry the three name-to-value dictionaries
                and the selected joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing, selects no joints by name, or if one of the
                three standard-deviation dictionaries is missing.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = [asset.joint_names[index] for index in joint_ids]
        self._std = {}
        for key in ("std_standing", "std_walking", "std_running"):
            if key not in cfg.params:
                raise ValueError(
                    f"The reward term '{self.__name__}' requires the '{key}' entry of its 'params'"
                    " to map joint-name patterns to standard deviations."
                )
            # ordered by the selected joints, so the vector lines up with ``joint_ids``
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            self._std[key] = torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float,
        running_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the posture reward.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            std_standing: Joint-name pattern to standard deviation [m or rad, depending on joint
                type] used below :attr:`walking_threshold`.
            std_walking: The same, used between the two thresholds.
            std_running: The same, used above :attr:`running_threshold`.
            walking_threshold: Command magnitude above which the robot counts as walking.
            running_threshold: Command magnitude above which the robot counts as running.
            asset_cfg: The articulation and the joints to hold at the stand pose.

        Returns:
            The reward in ``(0, 1]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        total_speed = _command_magnitude(env, command_name)

        standing_mask = (total_speed < walking_threshold).float().unsqueeze(1)
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float().unsqueeze(1)
        running_mask = (total_speed >= running_threshold).float().unsqueeze(1)
        std = (
            self._std["std_standing"] * standing_mask
            + self._std["std_walking"] * walking_mask
            + self._std["std_running"] * running_mask
        )

        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        error_squared = torch.square(joint_pos - default_joint_pos)
        return torch.exp(-torch.mean(error_squared / std**2, dim=1))


"""
Head and body pose tracking.
"""


def head_pose_tracking_raw(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    backlash_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking of the commanded head-pose delta, averaged over the neck and head joints.

    Args:
        env: The environment instance.
        command_name: Name of the head-pose command term. Its columns must line up with the joints
            selected by :attr:`asset_cfg`, which orders
            ``(neck_pitch, head_pitch, head_yaw, head_roll)``.
        std: Width of the per-joint Gaussian kernel [rad].
        asset_cfg: The articulation and the head joints to track. Select them by name with
            ``preserve_order=True`` so the columns match the command.
        backlash_cfg: Optional paired gearbox joints, added to the servo angles before scoring.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if backlash_cfg is not None:
        measured = measured + env.scene[backlash_cfg.name].data.joint_pos.torch[:, backlash_cfg.joint_ids]
    error = (measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]) - command
    return torch.exp(-((error / std) ** 2)).mean(dim=-1)


class head_pose_bias_penalty_gated(ManagerTermBase):
    """Penalize a head-pose error that persists, as opposed to one that averages out.

    An optional **upright gate** suppresses the term while the robot is low or tilted, for the
    recovery tasks whose episodes start on the ground. It multiplies both the error entering the
    average and the returned penalty, so a prone episode accumulates nothing and pays nothing --
    without it the term would tax the head-first phase of a stand-up, which is the motion it needs
    to discover. The velocity task leaves it off.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment moving average.

        Args:
            cfg: The term configuration, whose ``params`` carry the tracked joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing or selects no joints by name.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        self._error_ema = torch.zeros(env.num_envs, len(joint_ids), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the average for the environments that restarted.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._error_ema[:] = 0.0
        else:
            self._error_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tau_s: float,
        asset_cfg: SceneEntityCfg,
        gate_height_low: float | None = None,
        gate_height_high: float = 0.11,
        gate_tilt_full_deg: float = 20.0,
        gate_tilt_zero_deg: float = 45.0,
        backlash_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Advance the moving average and return the penalty.

        Args:
            env: The environment instance.
            command_name: Name of the head-pose command term.
            tau_s: Time constant of the moving average [s].
            asset_cfg: The articulation and the head joints to track. Mandatory: the term sizes its
                state from the selection at construction time, and the command columns are paired
                with the joints positionally.
            gate_height_low: Trunk height [m] below which the upright gate is fully closed. Defaults
                to None, which disables the gate entirely and leaves the other three unread.
            gate_height_high: Trunk height [m] above which the height half of the gate is fully
                open. Defaults to 0.11.
            gate_tilt_full_deg: Trunk tilt [deg] below which the tilt half of the gate is fully
                open. Defaults to 20.0.
            backlash_cfg: Optional paired gearbox joints, added before updating the error average.
            gate_tilt_zero_deg: Trunk tilt [deg] above which the tilt half of the gate is fully
                closed. Defaults to 45.0.

        Returns:
            The penalty in ``(-inf, 0]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        if backlash_cfg is not None:
            measured = measured + env.scene[backlash_cfg.name].data.joint_pos.torch[:, backlash_cfg.joint_ids]
        error = (measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]) - command

        gate = None
        if gate_height_low is not None:
            quat = asset.data.root_link_quat_w.torch
            gate = _height_gate(_root_height_above_ground(env, asset), gate_height_low, gate_height_high)
            gate = gate * _tilt_gate(quat, gate_tilt_full_deg, gate_tilt_zero_deg)
            error = error * gate.unsqueeze(-1)

        alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
        self._error_ema = (1.0 - alpha) * self._error_ema + alpha * error
        penalty = -self._error_ema.abs().mean(dim=-1)
        return penalty if gate is None else penalty * gate


def body_pose_tracking_6d(
    env: ManagerBasedRLEnv,
    command_name: str,
    nominal_height: float,
    xy_std: float,
    z_std: float,
    angle_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Reward tracking of a six-dimensional trunk-pose command, averaged over the six axes.

    There is no stock counterpart. :func:`isaaclab.envs.mdp.position_command_error` and
    :func:`~isaaclab.envs.mdp.orientation_command_error` measure a pose command as an unbounded
    error norm against a pose command term, not as a per-axis Gaussian around a nominal stand.

    Args:
        env: The environment instance.
        command_name: Name of the six-dimensional body-pose command term.
        nominal_height: Trunk height the vertical command is measured from [m].
        xy_std: Width of the Gaussian kernel on the horizontal position [m].
        z_std: Width of the Gaussian kernel on the height [m].
        angle_std: Width of the Gaussian kernel on each Euler angle [rad].
        asset_cfg: The articulation whose root link carries the trunk pose.
        axis_weights: Relative weight of each axis, ordered ``(x, y, z, roll, pitch, yaw)``. The
            reward is their weighted mean. Defaults to equal weights, which is the plain average.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    relative_pos = torch.nan_to_num(asset.data.root_link_pos_w.torch - env.scene.env_origins, nan=0.0)
    x_error = relative_pos[:, 0] - command[:, 0]
    y_error = relative_pos[:, 1] - command[:, 1]
    z_error = relative_pos[:, 2] - (nominal_height + command[:, 2])

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w.torch)
    roll_error = roll - command[:, 3]
    pitch_error = pitch - command[:, 4]
    yaw_error = math_utils.wrap_to_pi(yaw - command[:, 5])

    weight_x, weight_y, weight_z, weight_roll, weight_pitch, weight_yaw = axis_weights
    reward = weight_x * torch.exp(-((x_error / xy_std) ** 2))
    reward = reward + weight_y * torch.exp(-((y_error / xy_std) ** 2))
    reward = reward + weight_z * torch.exp(-((z_error / z_std) ** 2))
    reward = reward + weight_roll * torch.exp(-((roll_error / angle_std) ** 2))
    reward = reward + weight_pitch * torch.exp(-((pitch_error / angle_std) ** 2))
    reward = reward + weight_yaw * torch.exp(-((yaw_error / angle_std) ** 2))
    return reward / max(sum(axis_weights), 1e-6)


"""
Gait and feet.
"""


def feet_air_time_windowed_gated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float,
    gate_tilt_above_deg: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward feet whose current air time lies strictly inside a window, gated on the command.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the foot bodies to read. Select them by name with
            ``preserve_order=True``; the sensor resolves bodies in prim-label order.
        command_name: Name of the velocity command term.
        threshold_min: Lower edge of the rewarded air-time window [s], exclusive.
        threshold_max: Upper edge of the rewarded air-time window [s], exclusive.
        command_threshold: Command magnitude below which the reward is suppressed.
        gate_tilt_above_deg: Trunk tilt [deg] beyond which the reward is suppressed. Defaults to
            None, which pays a swinging foot whatever the trunk is doing.
        asset_cfg: The articulation whose root link carries the trunk. Read only by the gate.

    Returns:
        The number of feet inside the window, or zero for a standing command. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    reward = reward * (_command_magnitude(env, command_name) > command_threshold).float()
    if gate_tilt_above_deg is None:
        return reward
    asset: Articulation = env.scene[asset_cfg.name]
    return reward * (~_is_fallen(env, asset, gate_tilt_above_deg)).float()


def foot_origin_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize feet that deviate from a target height while moving horizontally.

    Args:
        env: The environment instance.
        target_height: Height above the ground the swinging foot should hold [m]. measures
            it at the sole site; this configuration measures the ankle body frame, so the value needs
            re-tuning by the sole-to-ankle offset rather than copying.
        command_name: Name of the velocity command term.
        asset_cfg: The articulation and the foot bodies to measure. Select them by name with
            ``preserve_order=True``.
        command_threshold: Command magnitude below which the penalty is suppressed.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = _feet_height_above_ground(env, asset_cfg)
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.abs(foot_height - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1)
    return cost * (_command_magnitude(env, command_name) > command_threshold).float()


class foot_origin_swing_height(ManagerTermBase):
    """Penalize, on touchdown, how far the peak of the swing was from a target height.

    Note:
        :meth:`~isaaclab.sensors.ContactSensor.compute_first_contact` returns a **float** mask of
        ones and zeros, not a boolean one, so its result is thresholded before it is used as a
        condition.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-foot peak heights.

        Args:
            cfg: The term configuration, whose ``params`` carry the tracked feet.
            env: The environment instance.

        Raises:
            ValueError: If ``sensor_cfg`` is missing from the term parameters.
        """
        super().__init__(cfg, env)

        sensor_cfg = _required_entity_cfg(cfg, "sensor_cfg", self.__name__)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        num_feet = len(contact_sensor.data.current_contact_time.torch[0, sensor_cfg.body_ids])
        self._peak_heights = torch.zeros(env.num_envs, num_feet, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the tracked peaks for the environments that restarted.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._peak_heights[:] = 0.0
        else:
            self._peak_heights[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        """Track the swing peaks and charge the landing ones.

        Args:
            env: The environment instance.
            sensor_cfg: The contact sensor and the foot bodies to read.
            asset_cfg: The articulation and the foot bodies to measure. Its body order must match
                :attr:`sensor_cfg`.
            target_height: Peak swing height the foot should reach [m]. Measured at the ankle body
                frame rather than at sole site.
            command_name: Name of the velocity command term.
            command_threshold: Command magnitude below which the penalty is suppressed.

        Returns:
            The cost in ``[0, inf)``. Shape is (num_envs,).
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_air = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        foot_height = _feet_height_above_ground(env, asset_cfg)
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, foot_height), self._peak_heights)

        # the sensor reports the landing mask as ones and zeros in float32, which cannot be used as
        # a ``torch.where`` condition directly
        first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        cost = cost * (_command_magnitude(env, command_name) > command_threshold).float()

        # the peak is charged once; the next swing starts from scratch
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return cost


def foot_slip_raw(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the squared planar speed of feet that are on the ground.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the foot bodies to read.
        command_name: Name of the velocity command term.
        asset_cfg: The articulation and the foot bodies to measure. Its body order must match
            :attr:`sensor_cfg`.
        command_threshold: Command magnitude below which the penalty is suppressed.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = (contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
    return cost * (_command_magnitude(env, command_name) > command_threshold).float()


"""
Trunk regularizers.
"""


def body_ang_vel_xy_l2_raw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the roll and pitch rate of selected bodies, in the world frame.

    The yaw rate is deliberately excluded: it is what the velocity command asks for.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the bodies to measure.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=(1, 2))


def angular_momentum_l2_raw(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the squared magnitude of the articulation's angular momentum about its own centre of mass.

    .. math::

        L = \\sum_i \\left[ R_i I_i R_i^T \\omega_i + m_i (r_i - r_c) \\times (v_i - v_c) \\right]

    where :math:`r_c` and :math:`v_c` are the mass-weighted centre of mass and its velocity. This
    is the same physical quantity the sensor reports, computed rather than measured. There is no
    stock counterpart.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the bodies to sum over. Defaults to every body of the
            entity named ``"robot"``, matching whole-robot subtree.

    Returns:
        The squared angular momentum [kg^2 m^4 / s^2]. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.data.body_mass.torch[:, asset_cfg.body_ids]
    pos_w = asset.data.body_com_pos_w.torch[:, asset_cfg.body_ids]
    lin_vel_w = asset.data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_com_ang_vel_w.torch[:, asset_cfg.body_ids]
    quat_w = asset.data.body_com_quat_w.torch[:, asset_cfg.body_ids]
    # ``body_inertia`` is stored about the body centre of mass in the body frame, so it has to be
    # rotated into the world frame before it can act on a world-frame angular velocity
    inertia_b = asset.data.body_inertia.torch[:, asset_cfg.body_ids].view(*mass.shape, 3, 3)

    total_mass = mass.sum(dim=1, keepdim=True).unsqueeze(-1)
    com_pos_w = (mass.unsqueeze(-1) * pos_w).sum(dim=1, keepdim=True) / total_mass
    com_lin_vel_w = (mass.unsqueeze(-1) * lin_vel_w).sum(dim=1, keepdim=True) / total_mass

    rot = math_utils.matrix_from_quat(quat_w)
    inertia_w = torch.matmul(rot, torch.matmul(inertia_b, rot.transpose(-2, -1)))
    spin = torch.matmul(inertia_w, ang_vel_w.unsqueeze(-1)).squeeze(-1)
    orbit = mass.unsqueeze(-1) * torch.cross(pos_w - com_pos_w, lin_vel_w - com_lin_vel_w, dim=-1)

    momentum = (spin + orbit).sum(dim=1)
    return torch.sum(torch.square(momentum), dim=-1)


def self_collision_cost_saturable(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0, saturate: bool = False
) -> torch.Tensor:
    """Penalize parts of the robot that are touching the robot itself.

    The sensor must be configured with
    :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` or its shape-level counterpart
    :attr:`~isaaclab.sensors.ContactSensorCfg.filter_shape_prim_expr` pointing back at the robot,
    otherwise no force matrix is produced.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the sensing objects to read.
        force_threshold: Contact-force magnitude above which a sensing object counts as touching [N].
            Defaults to 1.0, the same "a real contact" threshold the stock contact terms use.
        saturate: Whether to report 0-or-1 "is the robot touching itself" flag instead of
            a count. Defaults to False, which counts. Set it whenever the sensor senses *both* sides
            of a contact -- a many-to-many sensor reports one contact once per shape that carries
            it, so counting would scale the penalty with how finely the model is split into
            colliders rather than with how much of the robot is folded onto itself.

    Returns:
        Either the number of sensing objects in self-contact, or 0/1 when :attr:`saturate` is set.
        Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"The contact sensor '{sensor_cfg.name}' reports no force matrix. Set"
            " 'filter_prim_paths_expr' or 'filter_shape_prim_expr' on its configuration so that"
            " self-contacts are resolved."
        )
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
    touching = in_contact.any(dim=-1)
    return (touching.any(dim=-1) if saturate else touching.sum(dim=-1)).float()


"""
Standing up: posture, height and orientation terms.
"""


def joint_pose_gaussian(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward holding selected joints at the stand pose, with a single Gaussian tolerance.

    Args:
        env: The environment instance.
        std: Width of the per-joint Gaussian kernel [m or rad, depending on joint type].
        asset_cfg: The articulation and the joints to hold at the stand pose.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return torch.exp(-((error / std) ** 2)).mean(dim=-1)


def joint_pose_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the mean absolute deviation of selected joints from the stand pose.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the joints to hold at the stand pose.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return -torch.abs(error).mean(dim=-1)


def root_height_gaussian(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the trunk at a target height above the ground, using a Gaussian kernel.

    Args:
        env: The environment instance.
        target_height: Trunk height to hold [m], measured above the environment origin.
        std: Width of the Gaussian kernel [m].
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / std) ** 2))


def root_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the absolute distance of the trunk from a target height above the ground.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        target_height: Trunk height to hold [m], measured above the environment origin.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.abs(_root_height_above_ground(env, asset) - target_height)


def com_upward_velocity(
    env: ManagerBasedRLEnv,
    max_height: float,
    gate_tilt_above_deg: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward rising: the trunk's upward speed, paid only while it is below a ceiling.

    Args:
        env: The environment instance.
        max_height: Trunk height [m] above which the reward is switched off.
        gate_tilt_above_deg: Trunk tilt [deg] below which the reward is switched off. Defaults to
            None, which pays a rising robot at any tilt -- what the stand-up task, whose episodes
            all start on the ground, wants.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below_target = (_root_height_above_ground(env, asset) < max_height).float()
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    reward = torch.clamp(vertical_speed, min=0.0) * below_target
    if gate_tilt_above_deg is None:
        return reward
    return reward * _is_fallen(env, asset, gate_tilt_above_deg).float()


class trunk_vertical_accel_penalty(ManagerTermBase):
    """Penalize the magnitude of the trunk's vertical acceleration.

    Being a finite difference the term is stateful, so it is a class. The step after a reset is
    charged nothing: the previous velocity then belongs to the previous episode, and differencing
    across the discontinuity would charge every environment a spurious impulse on its first step.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the previous-velocity buffer.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._previous_speed = torch.zeros(env.num_envs, device=env.device)
        self._is_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the environments that restarted, so their next difference is not charged.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are marked.
        """
        if env_ids is None:
            self._is_fresh[:] = True
        else:
            self._is_fresh[env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Difference the vertical velocity and return the penalty.

        Args:
            env: The environment instance.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The penalty in ``(-inf, 0]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
        acceleration = (vertical_speed - self._previous_speed) / env.step_dt
        acceleration = torch.where(self._is_fresh, torch.zeros_like(acceleration), acceleration)
        self._previous_speed = vertical_speed.clone()
        self._is_fresh[:] = False
        return -torch.abs(acceleration)


def body_ang_vel_at_height(
    env: ManagerBasedRLEnv,
    height_low: float,
    height_high: float,
    tilt_full_deg: float,
    tilt_zero_deg: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the roll and pitch rate of a body, but only once the robot is up and near vertical.

    Args:
        env: The environment instance.
        height_low: Trunk height [m] below which the gate is fully closed.
        height_high: Trunk height [m] above which the height half of the gate is fully open.
        tilt_full_deg: Trunk tilt [deg] below which the tilt half of the gate is fully open.
        tilt_zero_deg: Trunk tilt [deg] above which the tilt half of the gate is fully closed.
        asset_cfg: The articulation and the bodies to measure.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(ang_vel_xy), dim=(1, 2))

    gate = _height_gate(_root_height_above_ground(env, asset), height_low, height_high)
    gate = gate * _tilt_gate(asset.data.root_link_quat_w.torch, tilt_full_deg, tilt_zero_deg)
    return cost * gate


def body_upright_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the cosine of the trunk's tilt from vertical.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[-1, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)


def upright_gaussian_at_height(
    env: ManagerBasedRLEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a near-vertical trunk, paid in proportion to how far the robot has risen.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel, in units of ``1 - cos(tilt)``.
        height_low: Trunk height [m] below which the gate is fully closed.
        height_high: Trunk height [m] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / std**2)
    return upright_score * _height_gate(_root_height_above_ground(env, asset), height_low, height_high)


def standing_composite_score(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the goal state as one multiplicative score over height, uprightness and pose.

    Args:
        env: The environment instance.
        target_height: Trunk height of the goal state [m].
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        pose_std: Width of the Gaussian kernel on the joint-position RMS error [rad].
        asset_cfg: The articulation and the joints scored against the stand pose. Its root link
            carries the trunk height and orientation.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_score = torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / height_std) ** 2))
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / upright_std**2)
    pose_error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    pose_score = torch.exp(-torch.square(pose_error).mean(dim=-1) / pose_std**2)
    return height_score * upright_score * pose_score


class joint_torque_rate_l2(ManagerTermBase):
    """Penalize the squared change in applied joint torque between two control steps.

    Being a finite difference the term is stateful, so it is a class.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the previous-torque buffer.

        Args:
            cfg: The term configuration, whose ``params`` carry the driven joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing or selects no joints by name.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        self._previous_torque = torch.zeros(env.num_envs, len(joint_ids), device=env.device)
        self._is_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the environments that restarted, so their next difference is not charged.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are marked.
        """
        if env_ids is None:
            self._is_fresh[:] = True
        else:
            self._is_fresh[env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Difference the applied torque and return the cost.

        Args:
            env: The environment instance.
            asset_cfg: The articulation and the driven joints. Mandatory: the term sizes its state
                from the selection at construction time.

        Returns:
            The cost in ``[0, inf)``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
        rate = torque - self._previous_torque
        rate = torch.where(self._is_fresh.unsqueeze(-1), torch.zeros_like(rate), rate)
        self._previous_torque = torque.clone()
        self._is_fresh[:] = False
        return torch.sum(torch.square(rate), dim=1)


"""
Potential-based progress. Both terms pay the *change* in a scalar potential and nothing for holding
any pose, which prevents stationary reward farming on a task whose episodes survive a fall.
Potential-based shaping is policy-invariant: it changes the learning signal while preserving the
optimal policy.
"""


class _PotentialProgress(ManagerTermBase):
    """Shared machinery of the two potential-based recovery terms."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the previous-potential buffer.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._is_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the environments that restarted, so their next difference is not paid or charged.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are marked.
        """
        if env_ids is None:
            self._is_fresh[:] = True
        else:
            self._is_fresh[env_ids] = True

    def _advance(self, potential: torch.Tensor) -> torch.Tensor:
        """Difference the potential against the previous step and re-baseline the fresh environments.

        Args:
            potential: This step's potential. Shape is (num_envs,).

        Returns:
            The change in potential, zero on the step after a reset. Shape is (num_envs,).
        """
        delta = potential - self._previous
        delta = torch.where(self._is_fresh, torch.zeros_like(delta), delta)
        self._previous = potential.clone()
        self._is_fresh[:] = False
        return delta


class upright_progress(_PotentialProgress):
    """Reward the increase in ``cos(tilt)`` of the trunk, and charge the decrease symmetrically.

    A full prone-to-standing recovery collects a total of about +1 before weighting, because
    ``cos(tilt)`` runs from 0 lying down to 1 upright.

    :func:`body_upright_linear` is the *level* of the same quantity, which the stand-up specialist
    can afford because its episodes end when the robot is up; on a task that keeps walking after a
    recovery the level would be a standing subsidy.
    """

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Difference the trunk's ``cos(tilt)`` against the previous control step.

        Args:
            env: The environment instance.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The change in ``cos(tilt)``, in ``[-2, 2]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        cos_tilt = torch.nan_to_num(1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch), nan=1.0)
        return self._advance(cos_tilt)


class height_progress(_PotentialProgress):
    """Reward the increase in trunk height below a ceiling, and charge the decrease symmetrically."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ceiling: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Difference the clamped trunk height against the previous control step.

        Args:
            env: The environment instance.
            ceiling: Trunk height [m] above which the potential is constant.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The change in clamped height [m]. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        return self._advance(torch.clamp(_root_height_above_ground(env, asset), max=ceiling))


"""
Recovery economics: the flat tax on staying down and the one-shot bounty for getting back up.
"""


class fallen_state_penalty(ManagerTermBase):
    """Charge a flat cost for every step spent fallen, until the stand is actually finished.

    The term returns a **positive** magnitude, so it is configured with a negative weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment armed latch.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._is_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Disarm the environments that restarted, so a new episode inherits no tax.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are disarmed.
        """
        if env_ids is None:
            self._is_armed[:] = False
        else:
            self._is_armed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gate_tilt_above_deg: float,
        release_tilt_below_deg: float | None = None,
        release_z_above: float | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Update the latch and return the cost.

        Args:
            env: The environment instance.
            gate_tilt_above_deg: Trunk tilt [deg] beyond which a fall arms the tax.
            release_tilt_below_deg: Trunk tilt [deg] below which the tax is released. Defaults to
                None, which drops the hysteresis and charges the instantaneous fallen state.
            release_z_above: Trunk height [m] above which the tax is released, on top of the tilt
                condition. Defaults to None, which releases on tilt alone.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The cost, 1.0 while armed and 0.0 otherwise. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        fallen = _is_fallen(env, asset, gate_tilt_above_deg)
        if release_tilt_below_deg is None:
            return fallen.float()

        cos_tilt = 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)
        recovered = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
        if release_z_above is not None:
            recovered &= _root_height_above_ground(env, asset) > release_z_above
        self._is_armed |= fallen
        self._is_armed &= ~recovered
        return self._is_armed.float()


class recovery_success(ManagerTermBase):
    """Pay a one-shot bounty the step a robot finishes standing up after a genuine fall.

    Two guards make it a *recovery* bounty rather than a standing subsidy: the robot must have been
    fallen continuously for :attr:`min_fallen_s` before the bounty arms, so a gait wobble past the
    tilt bound cannot claim it, and the latch clears when it fires.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the fallen clock and the armed latch.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._fallen_s = torch.zeros(env.num_envs, device=env.device)
        self._is_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the clock and the latch for the environments that restarted.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._fallen_s[:] = 0.0
            self._is_armed[:] = False
        else:
            self._fallen_s[env_ids] = 0.0
            self._is_armed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        fallen_tilt_deg: float,
        min_fallen_s: float,
        up_tilt_deg: float,
        up_z: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Advance the fallen clock and return the bounty on the step a recovery completes.

        Args:
            env: The environment instance.
            fallen_tilt_deg: Trunk tilt [deg] beyond which the fallen clock runs.
            min_fallen_s: Continuous time [s] spent fallen before the bounty arms.
            up_tilt_deg: Trunk tilt [deg] below which the robot counts as recovered.
            up_z: Trunk height [m] above which it counts as recovered, on top of the tilt condition.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The bounty, 1.0 on the step a recovery completes and 0.0 otherwise. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        cos_tilt = 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)
        fallen = _is_fallen(env, asset, fallen_tilt_deg)
        recovered = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (_root_height_above_ground(env, asset) > up_z)

        self._fallen_s = torch.where(fallen, self._fallen_s + env.step_dt, torch.zeros_like(self._fallen_s))
        self._is_armed |= self._fallen_s >= min_fallen_s
        fired = self._is_armed & recovered
        self._is_armed &= ~fired
        return fired.float()


"""
Rolling forward (roulade): the rotation accumulator and the terms that read it.
"""

_ROULADE_FORWARD_SIGN = 1.0

"""Sign of the body-frame pitch rate that counts as *forward* rotation.

Face down is a +90 degree pitch in this family's reset convention, so a forward roll is a positive
body-frame ``omega_y``.
"""

_ROULADE_HEAD_LATCH_WINDOW = (math.radians(20.0), math.radians(170.0))

"""Rotation window [rad] inside which head-ground contact latches the roll as over-the-head.

In a real roulade the head plants at 60 to 120 degrees of body rotation; the window is generous
around that, so contact before the robot has committed and contact once it is already on its back
do not count.
"""

_ROULADE_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)

"""The flat top of the head shell, as a unit vector in the head body's own frame."""

_ROULADE_HEAD_TOP_DOWN_MIN = 0.3

"""How far the head top must point at the floor before a head contact counts, as ``dot(axis, -z)``."""

_ROULADE_SAGITTAL_FULL = 0.5

_ROULADE_SAGITTAL_ZERO = 0.866

"""Bounds on ``|lateral axis z|`` between which rotation credit fades from full to none."""


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z component of a body's lateral (y) axis. Zero when the body is sagittally flat.

    Args:
        quat: Body orientation in (x, y, z, w). Shape is (num_envs, 4).

    Returns:
        The lateral axis' world-z component in ``[-1, 1]``. Shape is (num_envs,).
    """
    return 2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 3] * quat[:, 0])


def _head_top_is_down(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Whether the flat top of the head shell is pointing at the floor.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the single body carrying the head shells.

    Returns:
        Whether the head top faces down. Shape is (num_envs,).

    Raises:
        ValueError: If ``asset_cfg`` does not select exactly one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None or len(asset_cfg.body_ids) != 1:
        raise ValueError(
            "The roll's head terms measure the orientation of a single head body; 'asset_cfg' must"
            f" select exactly one by name. Received: {asset_cfg.body_names}."
        )
    head_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    axis_b = torch.tensor(_ROULADE_HEAD_TOP_AXIS, device=head_quat_w.device).expand_as(head_quat_w[..., :3])
    axis_world_z = math_utils.quat_apply(head_quat_w, axis_b).squeeze(1)[:, 2]
    return axis_world_z < -_ROULADE_HEAD_TOP_DOWN_MIN


def _any_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, filtered: bool = False) -> torch.Tensor:
    """Whether any object a contact sensor senses is touching something.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the sensing objects to read.
        filtered: Whether to read the per-partner force matrix rather than the net force. Defaults to
            False. Set it when the question is "touching *that*" rather than "touching anything" --
            the net force sums every contact a sensing object carries, including the robot's own
            limbs, so only the matrix can answer it.

    Returns:
        Whether at least one selected sensing object is in contact. Shape is (num_envs,).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not filtered:
        forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
        return (forces.norm(dim=-1) > 0.0).any(dim=-1)
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"The contact sensor '{sensor_cfg.name}' reports no force matrix. Set"
            " 'filter_prim_paths_expr' or 'filter_shape_prim_expr' on its configuration so that its"
            " contact partners are resolved."
        )
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > 0.0).any(dim=-1).any(dim=-1)


def _update_roulade_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg,
    head_sensor_cfg: SceneEntityCfg,
) -> None:
    """Advance the rotation accumulator and the head latch by one control step.

    The frontier only moves forward, so rocking backwards neither pays nor un-pays, and the head
    latch is set once head-ground contact happens with the flat top of the head pointing down while
    the accumulated angle is inside :data:`_ROULADE_HEAD_LATCH_WINDOW`.

    The update is guarded by the global step count, so calling it from several terms in one control
    step integrates once.

    Args:
        env: The environment instance.
        asset_cfg: The articulation, whose root link carries the trunk and whose single selected
            body carries the head shells.
        support_sensor_cfg: The contact sensor reporting whether the robot touches the ground. It
            must be filtered against the terrain, because the gate reads its force matrix.
        head_sensor_cfg: The contact sensor reporting whether the head touches the ground.
    """
    state = roulade_roll_state(env)
    step = int(env.common_step_counter)
    if step == state.last_update_step:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    forward_rate = _ROULADE_FORWARD_SIGN * asset.data.root_link_ang_vel_b.torch[:, 1]
    delta = torch.nan_to_num(forward_rate, nan=0.0) * env.step_dt
    delta = delta * _any_contact(env, support_sensor_cfg, filtered=True).float()
    lateral = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w.torch), nan=1.0).abs()
    sagittal_gate = _smoothstep((_ROULADE_SAGITTAL_ZERO - lateral) / (_ROULADE_SAGITTAL_ZERO - _ROULADE_SAGITTAL_FULL))
    # in place throughout, so the buffers stay the ones allocated on the environment: a rollout runs
    # under ``torch.inference_mode`` and rebinding them there would leave the reset event unable to
    # write them from outside it
    state.accumulated_angle.add_(delta * sagittal_gate)
    torch.maximum(state.frontier, state.accumulated_angle, out=state.frontier)

    latch_lo, latch_hi = _ROULADE_HEAD_LATCH_WINDOW
    in_window = (state.accumulated_angle > latch_lo) & (state.accumulated_angle < latch_hi)
    head_contact = _any_contact(env, head_sensor_cfg)
    state.head_latch.logical_or_(head_contact & in_window & _head_top_is_down(env, asset_cfg))
    state.last_update_step = step


def roulade_completion_gate(
    env: ManagerBasedRLEnv, gate_lo: float, gate_hi: float, require_head: bool = True
) -> torch.Tensor:
    """Smoothstep on the rotation frontier: zero below ``gate_lo``, one above ``gate_hi``.

    It is public because it is the whole landing annuity in one function: a new gated roll reward
    multiplies by this and nothing else, and a test that wants to know whether completion is
    *reachable* reads it directly.

    Args:
        env: The environment instance.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        require_head: Whether the gate additionally requires the over-the-head latch. Defaults to
            True, which is what every shipped landing reward passes -- "went over the head" is a
            requirement of the trick rather than a bonus.

    Returns:
        The gate in ``[0, 1]``. Shape is (num_envs,).
    """
    state = roulade_roll_state(env)
    gate = _smoothstep((state.frontier - gate_lo) / max(gate_hi - gate_lo, 1e-6))
    if require_head:
        gate = gate * state.head_latch.float()
    return gate


def roulade_progress(
    env: ManagerBasedRLEnv,
    target_angle: float,
    max_paid_rate: float,
    support_sensor_cfg: SceneEntityCfg,
    head_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Pay increments of the rotation frontier, up to one full roll, at a capped rate.

    The rate cap **forfeits** rather than defers: the paid pointer still jumps to the frontier, so a
    violent whip collects strictly *less* total progress reward than a controlled roll rather than
    the same amount sooner.

    This is also the term that advances the accumulator every control step -- see
    :func:`_update_roulade_state`. The other roll terms only read the frontier it leaves behind, so
    it has to be **declared first** among them; Isaac Lab evaluates reward terms in declaration
    order.

    Args:
        env: The environment instance.
        target_angle: Rotation [rad] that counts as a complete roll, normally ``2 pi``.
        max_paid_rate: Largest rotation rate [rad/s] that is paid for. Rotation faster than this
            forfeits the excess.
        support_sensor_cfg: The contact sensor reporting whether the robot touches the ground, which
            gates the accumulator. It must be filtered against the terrain.
        head_sensor_cfg: The contact sensor reporting whether the head touches the ground, which
            drives the over-the-head latch.
        asset_cfg: The articulation, whose root link carries the trunk and whose single selected
            body carries the head shells.

    Returns:
        The reward in ``[0, inf)``, normalized so a roll at the cap pays about one per step.
        Shape is (num_envs,).
    """
    _update_roulade_state(env, asset_cfg, support_sensor_cfg, head_sensor_cfg)
    state = roulade_roll_state(env)
    new_paid = torch.clamp(state.frontier, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(state.paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    torch.maximum(state.paid, new_paid, out=state.paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    angle_lo: float,
    angle_hi: float,
    rate_norm: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward pivoting over the head: head-ground contact while rotating forward, mid-roll.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor reporting whether the head touches the ground.
        angle_lo: Rotation [rad] below which the term is off.
        angle_hi: Rotation [rad] above which the term is off.
        rate_norm: Forward rotation rate [rad/s] at which the rate factor saturates.
        asset_cfg: The articulation, whose root link carries the trunk and whose single selected
            body carries the head shells.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    state = roulade_roll_state(env)
    contact = _any_contact(env, sensor_cfg).float()
    in_window = ((state.accumulated_angle > angle_lo) & (state.accumulated_angle < angle_hi)).float()
    forward_rate = _ROULADE_FORWARD_SIGN * asset.data.root_link_ang_vel_b.torch[:, 1]
    rate = torch.clamp(torch.nan_to_num(forward_rate, nan=0.0) / rate_norm, 0.0, 1.0)
    head_top = 0.3 + 0.7 * _head_top_is_down(env, asset_cfg).float()
    return contact * in_window * rate * head_top


def roulade_landing_composite(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward standing at the stand pose, but only once the roll has been completed over the head.

    Args:
        env: The environment instance.
        target_height: Trunk height of the goal state [m].
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        pose_std: Width of the Gaussian kernel on the joint-position RMS error [rad].
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation and the joints scored against the stand pose. Its root link
            carries the trunk height and orientation.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        asset_cfg=asset_cfg,
    )
    return score * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_upright_after_roll(
    env: ManagerBasedRLEnv,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a vertical trunk once the roll is complete, linearly in ``cos(tilt)``.

    Args:
        env: The environment instance.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)
    return torch.clamp(upright_score, min=0.0) * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_height_after_roll(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward standing height once the roll is complete, with a broad Gaussian.

    Args:
        env: The environment instance.
        target_height: Trunk height to reach [m], measured above the environment origin.
        std: Width of the Gaussian kernel [m].
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_score = torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / std) ** 2))
    return height_score * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_landing_sharp(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the last mile of the landing: tight Gaussians on height and tilt, behind the gate.

    Args:
        env: The environment instance.
        target_height: Trunk height of the goal state [m].
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / upright_std**2)
    height_score = torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / height_std) ** 2))
    return upright_score * height_score * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_stand_tax(
    env: ManagerBasedRLEnv,
    target_height: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Charge every step spent below standing height once the roll is complete.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        target_height: Trunk height [m] below which the shortfall is charged.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    shortfall = torch.clamp(target_height - _root_height_above_ground(env, asset), min=0.0)
    return -shortfall * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_rise_velocity(
    env: ManagerBasedRLEnv,
    max_height: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward rising out of the roll, gated to its late phase.

    Args:
        env: The environment instance.
        max_height: Trunk height [m] above which the reward is switched off.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below_ceiling = (_root_height_above_ground(env, asset) < max_height).float()
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    reward = torch.clamp(vertical_speed, min=0.0) * below_ceiling
    return reward * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_overspeed_penalty(
    env: ManagerBasedRLEnv, omega_max: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize rotating faster than a threshold, quadratically in the excess.

    Args:
        env: The environment instance.
        omega_max: Forward rotation rate [rad/s] above which the excess is charged.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rate = torch.nan_to_num(asset.data.root_link_ang_vel_b.torch[:, 1], nan=0.0)
    return torch.clamp(rate.abs() - omega_max, min=0.0).pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize tipping out of the sagittal plane, quadratically in the lateral axis' world-z.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w.torch), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize body-frame roll and yaw rate, leaving the roll axis free.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rate = asset.data.root_link_ang_vel_b.torch
    return torch.nan_to_num(rate[:, 0].pow(2) + rate[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize sideways travel, quadratically in the body-frame lateral velocity.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b.torch[:, 1].pow(2), nan=0.0)


"""
Roller skating: the wheel, gait and lean terms.
"""


def com_height_target(
    env: ManagerBasedRLEnv,
    target_height_min: float,
    target_height_max: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the trunk inside a height band, and charge the squared miss outside it.

    Args:
        env: The environment instance.
        target_height_min: Lower edge of the rewarded band [m], above the environment origin.
        target_height_max: Upper edge of the rewarded band [m].
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``(-inf, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = _root_height_above_ground(env, asset)
    below = height < target_height_min
    above = height > target_height_max
    penalty = torch.square(height - target_height_min) * below.float()
    penalty += torch.square(height - target_height_max) * above.float()
    return (~(below | above)).float() - penalty


def feet_flat_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    normal_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    bodies_per_foot: int = 1,
) -> torch.Tensor:
    """Penalize a foot whose sole is not parallel to the ground.

    Note:
        The tilt is NaN-guarded, so a broken environment is charged nothing rather than poisoning the
        batch. A rare solver divergence leaves the body orientations non-finite for a single step,
        and no termination can save the reward for that step: ``ManagerBasedRLEnv.step`` computes
        both the terminations and the rewards from the same post-physics buffers and only *resets*
        the flagged environments afterwards, in ``_reset_idx``. So ``nan_state`` does detect the
        divergence and does recycle the environment, but the reward for the step it happened on is
        computed on the poisoned state either way -- and RSL-RL aborts training on a NaN reward. The
        guard is a no-op on any finite orientation.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the foot bodies to measure, in the sensor's foot order.
        sensor_cfg: The contact sensor and the bodies whose contact gates each foot. Defaults to
            None, which charges every selected foot whether or not it is loaded.
        normal_axis: Sole normal in the foot body frame [-]. Defaults to the body ``z`` axis.
        bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    foot_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity_dir_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    gravity_dir_w = gravity_dir_w.unsqueeze(1).expand(-1, foot_quat_w.shape[1], -1)
    gravity_dir_b = math_utils.quat_apply_inverse(foot_quat_w, gravity_dir_w)
    normal = torch.tensor(normal_axis, dtype=gravity_dir_b.dtype, device=gravity_dir_b.device)
    normal = torch.nn.functional.normalize(normal, dim=-1)

    tilt = 1.0 - torch.square(torch.sum(gravity_dir_b * normal, dim=-1))
    # A diverged solver leaves the body orientations non-finite for exactly one step, and the reset
    # that repairs it runs after the reward is computed, so no termination can keep the NaN out of
    # the reward buffer RSL-RL checks. A blade whose frame is not a number is not tilted in any
    # measurable sense, so the guard charges nothing; it is a no-op on any finite orientation. Same
    # guard, same reason, as ``wheel_glide_reward``.
    tilt = torch.nan_to_num(tilt, nan=0.0, posinf=0.0, neginf=0.0)

    if sensor_cfg is None:
        return torch.sum(tilt, dim=1)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = _observations.fold_bodies_into_feet(
        contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids], bodies_per_foot
    ).amax(dim=2)
    return torch.sum(tilt * (contact_time > 0.0).float(), dim=1)


class joint_action_rate_l2(ManagerTermBase):
    """Penalize the squared change of the raw actions driving a subset of the joints."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve the action columns the selected joints are driven through.

        Args:
            cfg: The term configuration, whose ``params`` carry the action term and the joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing, selects no joints by name, or selects a joint
                the named action term does not drive.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        asset: Articulation = env.scene[asset_cfg.name]
        action_name = cfg.params["action_name"]
        action_term = env.action_manager.get_term(action_name)
        driven_ids = [int(index) for index in action_term.joint_ids]
        columns = []
        for joint_id in joint_ids:
            if joint_id not in driven_ids:
                raise ValueError(
                    f"The reward term '{self.__name__}' selects joint '{asset.joint_names[joint_id]}', which the"
                    f" action term '{action_name}' does not drive."
                )
            columns.append(driven_ids.index(joint_id))
        # the term's slice of the concatenated action vector the manager stores
        active_terms = list(env.action_manager.active_terms)
        start = sum(env.action_manager.action_term_dim[: active_terms.index(action_name)])
        self._columns = torch.tensor(columns, dtype=torch.long, device=env.device) + start

    def __call__(self, env: ManagerBasedRLEnv, action_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Difference the selected action columns and return the cost.

        Args:
            env: The environment instance.
            action_name: Name of the joint-position action term. Mandatory: the term resolves its
                columns from it at construction time.
            asset_cfg: The articulation and the joints to charge. Mandatory, and selected by name.

        Returns:
            The cost in ``[0, inf)``. Shape is (num_envs,).
        """
        del action_name, asset_cfg
        rate = env.action_manager.action[:, self._columns] - env.action_manager.prev_action[:, self._columns]
        return torch.sum(torch.square(rate), dim=1)


def joint_pose_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the squared deviation of selected joints from the stand pose.

    Note:
        The deviation is NaN-guarded, so a broken environment scores zero rather than poisoning the
        batch. A rare solver divergence leaves the joint state non-finite for a single step, and no
        termination can save the reward for that step: ``ManagerBasedRLEnv.step`` computes both the
        terminations and the rewards from the same post-physics buffers and only *resets* the
        flagged environments afterwards, in ``_reset_idx``. So ``nan_state`` does detect the
        divergence and does recycle the environment, but the reward for the step it happened on is
        computed on the poisoned state either way -- and RSL-RL aborts training on a NaN reward. The
        guard is a no-op on any finite state.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the joints to hold at the stand pose.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    # A diverged solver leaves the whole joint state non-finite for exactly one step, and the reset
    # that repairs it runs after the reward is computed, so no termination can keep the NaN out of
    # the reward buffer RSL-RL checks. Sanitizing the *error* rather than the sum keeps that
    # environment's contribution at zero -- it has no pose to be away from -- and is a no-op on any
    # finite state. Same guard, same reason, as ``wheel_glide_reward``.
    error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sum(torch.square(error), dim=1)


def action_over_limit_penalty(
    env: ManagerBasedRLEnv,
    action_name: str,
    overshoot: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize a joint *command* driven past its hard stop by more than a tolerance.

    Args:
        env: The environment instance.
        action_name: Name of the joint-position action term whose target is charged.
        overshoot: Tolerance [rad] the command may exceed a hard stop by before it is charged.
        asset_cfg: The articulation the action term drives.

    Returns:
        The total overshoot [rad] in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)

    target = term.processed_actions
    limits = asset.data.joint_pos_limits.torch[:, term.joint_ids]
    over = (target - (limits[..., 1] + overshoot)).clamp(min=0.0)
    over += ((limits[..., 0] - overshoot) - target).clamp(min=0.0)
    return torch.sum(over, dim=-1)


def wheel_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    vel_scale: float,
    wheel_radius: float,
    bidirectional: bool = False,
) -> torch.Tensor:
    """Reward spinning the passive wheels forward, in proportion to the commanded throttle.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term whose first column is the throttle.
        asset_cfg: The articulation and the passive wheel joints to average.
        vel_scale: Ground speed [m/s] the ``tanh`` is scaled to saturate near.
        wheel_radius: Rolling radius [m] the ground speed is converted to a wheel rate with.
        bidirectional: Whether a negative throttle pays for spinning backwards. Defaults to False,
            which is what ships.

    Returns:
        The reward in ``[0, |cmd_x|]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    forward_omega = torch.mean(asset.data.joint_vel.torch[:, asset_cfg.joint_ids], dim=1)
    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        aligned = torch.sign(command_x) * forward_omega
        return torch.abs(command_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(command_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def wheel_glide_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    cap_speed: float = 0.35,
    wheel_radius: float = 0.0175,
) -> torch.Tensor:
    """Reward letting the passive wheels roll forward, up to a cap.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the passive wheel joints to average. On the roller model the
            wheels interleave with the servos, so they are selected by name rather than by index.
        cap_speed: Rolling speed [m/s] the reward saturates at. Defaults to 0.35,
        wheel_radius: Rolling radius [m] the wheel rate is converted to a ground speed with. Defaults
            to 0.0175,

    Returns:
        The reward in ``[0, cap_speed]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forward_omega = torch.mean(asset.data.joint_vel.torch[:, asset_cfg.joint_ids], dim=1)
    # A rare contact divergence turns a free-spinning wheel's rate into a NaN, and this task carries
    # no reward-side NaN policy to fall back on.
    speed = torch.nan_to_num(forward_omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def braking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    vel_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward coming to a stop, in proportion to a negative throttle.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term whose first column is the throttle.
        vel_std: Width [m/s] of the Gaussian on the residual forward speed.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, |cmd_x|]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    braking_strength = torch.clamp(-env.command_manager.get_command(command_name)[:, 0], min=0.0)
    forward_vel = asset.data.root_link_lin_vel_b.torch[:, 0]
    stopped = torch.exp(-torch.square(forward_vel.clamp(min=0.0)) / vel_std**2)
    return braking_strength * stopped


def _forward_progress_gate(
    env: ManagerBasedRLEnv, vel_ref: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor | None:
    """``clamp(v_fwd, min=0) / v_ref`` gate, clipped to 1 and None when disabled."""
    if vel_ref <= 0.0:
        return None
    asset: Articulation = env.scene[asset_cfg.name]
    forward_vel = asset.data.root_link_lin_vel_b.torch[:, 0]
    return (forward_vel.clamp(min=0.0) / vel_ref).clamp(max=1.0)


def skating_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold_min: float,
    threshold_max: float,
    vel_gate_ref: float,
    bodies_per_foot: int = 1,
) -> torch.Tensor:
    """Reward feet whose air time lies inside a window, scaled by throttle and by forward progress.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read, in per-foot order. The sensor must
            track air time.
        command_name: Name of the velocity command term whose first column is the throttle.
        threshold_min: Lower edge of the rewarded air-time window [s], exclusive.
        threshold_max: Upper edge of the rewarded air-time window [s], exclusive.
        vel_gate_ref: Forward speed [m/s] at which the progress gate is fully open.
        bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

    Returns:
        The reward in ``[0, num_feet * cmd_x]``. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = _observations.fold_bodies_into_feet(
        contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids], bodies_per_foot
    ).amin(dim=2)
    in_range = (air_time > threshold_min) & (air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    reward = reward * torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    return reward if gate is None else reward * gate


def _feet_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, bodies_per_foot: int) -> torch.Tensor:
    """Number of feet currently loaded, from the sensor's per-body contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = _observations.fold_bodies_into_feet(
        contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids], bodies_per_foot
    ).amax(dim=2)
    return torch.sum((contact_time > 0.0).float(), dim=1)


def single_support_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    vel_gate_ref: float,
    double_penalty: float = 0.25,
    bodies_per_foot: int = 1,
) -> torch.Tensor:
    """Reward standing on exactly one foot while pushing, and charge standing on both.

    The double-support charge is **not** speed-gated, deliberately, so it also applies to a robot
    that is commanded forward and standing still on both feet.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read, in per-foot order.
        command_name: Name of the velocity command term whose first column is the throttle.
        vel_gate_ref: Forward speed [m/s] at which the progress gate is fully open.
        double_penalty: Fraction of the throttle charged for standing on both feet. Defaults to 0.25,
            which is default.
        bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

    Returns:
        The reward in ``[-double_penalty * cmd_x, cmd_x]``. Shape is (num_envs,).
    """
    num_in_contact = _feet_in_contact(env, sensor_cfg, bodies_per_foot)
    single = (num_in_contact == 1).float()
    double = (num_in_contact >= 2).float()

    command_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_reward = single * command_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_reward = single_reward * gate
    return single_reward - double_penalty * double * command_x


def glide_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    vel_ref: float,
    stillness_std: float = 5.0,
    bodies_per_foot: int = 1,
) -> torch.Tensor:
    """Reward coasting on one foot with quiet legs, in proportion to the forward progress.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read, in per-foot order.
        command_name: Name of the velocity command term whose first column is the throttle.
        asset_cfg: The articulation and the leg joints whose stillness is measured.
        vel_ref: Forward speed [m/s] at which the progress gate is fully open.
        stillness_std: Width [rad/s] of the Gaussian on the summed squared leg joint velocity.
            Defaults to 5.0, which is default.
        bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    single = (_feet_in_contact(env, sensor_cfg, bodies_per_foot) == 1).float()

    gate = _forward_progress_gate(env, vel_ref)
    if gate is None:
        gate = torch.ones(env.num_envs, device=env.device)

    joint_vel_squared = torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_squared / stillness_std**2)
    active = (env.command_manager.get_command(command_name)[:, 0] >= 0.0).float()
    return single * gate * stillness * active


class gait_symmetry_penalty(ManagerTermBase):
    """Penalize an episode that has spent much more swing time on one foot than on the other."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-foot swing-time accumulator.

        Args:
            cfg: The term configuration, whose ``params`` carry the tracked feet.
            env: The environment instance.

        Raises:
            ValueError: If ``sensor_cfg`` is missing, or if the feet do not come in a left/right
                pair -- the imbalance is defined between exactly two of them.
        """
        super().__init__(cfg, env)

        sensor_cfg = _required_entity_cfg(cfg, "sensor_cfg", self.__name__)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        num_bodies = len(contact_sensor.data.current_air_time.torch[0, sensor_cfg.body_ids])
        num_feet = num_bodies // int(cfg.params.get("bodies_per_foot", 1))
        if num_feet != 2:
            raise ValueError(
                f"The reward term '{self.__name__}' compares the swing time of a left and a right"
                f" foot; 'sensor_cfg' resolved to {num_feet} feet."
            )
        self._swing_time = torch.zeros(env.num_envs, num_feet, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the accumulated swing time of the environments that restarted.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._swing_time[:] = 0.0
        else:
            self._swing_time[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, bodies_per_foot: int = 1) -> torch.Tensor:
        """Accumulate this step's swing time and return the normalized imbalance.

        Args:
            env: The environment instance.
            sensor_cfg: The contact sensor and the bodies to read, in per-foot order. Mandatory: the
                term sizes its state from the selection at construction time.
            bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

        Returns:
            The cost in ``[0, 1)``. Shape is (num_envs,).
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        air_time = _observations.fold_bodies_into_feet(
            contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids], bodies_per_foot
        ).amin(dim=2)
        self._swing_time += (air_time > 0.0).float() * env.step_dt
        left, right = self._swing_time[:, 0], self._swing_time[:, 1]
        return torch.abs(left - right) / (left + right + 1e-3)


def forward_lean_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pitch: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward leaning into the push, in proportion to the commanded throttle.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term whose first column is the throttle.
        target_pitch: Rewarded value [-] of the forward gravity component, i.e. ``sin`` of the lean.
        std: Width [-] of the Gaussian around it.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, cmd_x]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forward_lean = asset.data.projected_gravity_b.torch[:, 0]
    push = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    return push * torch.exp(-torch.square(forward_lean - target_pitch) / std**2)


class heading_hold_reward(ManagerTermBase):
    """Reward holding the heading the episode started with.

    Being anchored to a per-episode reference heading it is stateful, so it is a class. There is no
    stock counterpart: :func:`isaaclab.envs.mdp.heading_command_error_abs` tracks a *commanded*
    heading, which this task has none of.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment reference heading.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._reference_heading = torch.zeros(env.num_envs, device=env.device)
        self._is_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the environments that restarted, so their next call re-anchors the reference.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are marked.
        """
        if env_ids is None:
            self._is_fresh[:] = True
        else:
            self._is_fresh[env_ids] = True

    def __call__(
        self, env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """Anchor the reference on the first call of an episode and score the drift from it.

        Args:
            env: The environment instance.
            std: Width [rad] of the Gaussian on the heading error.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The reward in ``(0, 1]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        heading = asset.data.heading_w.torch
        # Two NaN guards, both no-ops on a finite heading, both for the divergence
        # :func:`joint_pose_l2` documents. The first keeps a heading that is not a number out of the
        # *anchor*, which would otherwise persist for the rest of the episode; the second scores an
        # unmeasurable heading as maximally wrong -- half a turn, the far end of the wrapped range --
        # so a broken environment earns nothing instead of full marks.
        self._reference_heading = torch.where(
            self._is_fresh, torch.nan_to_num(heading, nan=0.0, posinf=0.0, neginf=0.0), self._reference_heading
        )
        self._is_fresh[:] = False
        error = math_utils.wrap_to_pi(heading - self._reference_heading)
        error = torch.nan_to_num(error, nan=math.pi, posinf=math.pi, neginf=math.pi)
        return torch.exp(-torch.square(error) / std**2)


def leg_symmetry_reward(
    env: ManagerBasedRLEnv, left_joint_cfg: SceneEntityCfg, right_joint_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward the two legs mirroring each other, which is the swizzle's defining symmetry.

    The condition is ``q_left + q_right ~= 0``, not ``q_left ~= q_right``, because the model uses
    mirrored left/right sign conventions: a symmetric pose reads as equal and opposite joint angles.
    Compare :func:`leg_antisymmetry`, the spin task's term, which asks for the other one.

    Args:
        env: The environment instance.
        left_joint_cfg: The articulation and the left-leg joints, in the order they pair up.
        right_joint_cfg: The articulation and the right-leg joints, in the matching order.

    Returns:
        The reward in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[left_joint_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    residual = joint_pos[:, left_joint_cfg.joint_ids] + joint_pos[:, right_joint_cfg.joint_ids]
    return -torch.abs(residual).mean(dim=-1)


def grounded_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, bodies_per_foot: int = 1
) -> torch.Tensor:
    """Reward keeping both blades on the ground, in proportion to the commanded throttle magnitude.

    The scale is ``|cmd_x|`` rather than ``clamp(cmd_x, min=0)``, so it shapes the push in either
    direction -- this task's throttle range is symmetric.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read, in per-foot order.
        command_name: Name of the velocity command term whose first column is the throttle.
        bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

    Returns:
        The reward in ``[0, |cmd_x|]``. Shape is (num_envs,).
    """
    grounded = (_feet_in_contact(env, sensor_cfg, bodies_per_foot) >= 2.0).float()
    return grounded * torch.abs(env.command_manager.get_command(command_name)[:, 0])


def heading_tracking_reward(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward driving the commanded heading error to zero.

    Note:
        This term and :func:`heading_hold_reward` are in genuine tension while both weights are
        non-zero -- one pays for holding the spawn heading and the other for leaving it -- which is
        why the task hands over between them with a pair of crossing curricula rather than enabling
        both outright.

    Args:
        env: The environment instance.
        command_name: Name of the relative-heading command term whose third column is the error.
        std: Width [rad] of the Gaussian on the heading error.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    heading_error = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.square(heading_error) / std**2)


def _ball_forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Signed ball speed [m/s] along the episode's frozen kick direction."""
    ball: RigidObject = env.scene[asset_cfg.name]
    velocity_xy = ball.data.root_link_lin_vel_w.torch[:, :2]
    forward = (velocity_xy * ball_kick_direction(env)).sum(dim=1)
    return torch.nan_to_num(forward, nan=0.0, posinf=0.0, neginf=0.0)


def ball_forward_velocity(
    env: ManagerBasedRLEnv, max_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")
) -> torch.Tensor:
    """Reward the ball rolling away along the direction the robot was facing at reset.

    Args:
        env: The environment instance.
        max_speed: Ball speed [m/s] at which the reward saturates.
        asset_cfg: The rigid object whose linear velocity is read.

    Returns:
        The reward in ``[0, max_speed]``. Shape is (num_envs,).
    """
    return _ball_forward_speed(env, asset_cfg).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(
    env: ManagerBasedRLEnv,
    target_speed: float,
    max_penalty: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Charge the ball's forward speed above a target, up to a cap.

    The cap bounds the worst single step a wild kick can cost, which is what keeps one lucky
    smash from dominating a whole episode's return.

    Args:
        env: The environment instance.
        target_speed: Ball speed [m/s] above which the overshoot is charged.
        max_penalty: Largest overshoot [m/s] charged in one step. Defaults to 5.0, value.
        asset_cfg: The rigid object whose linear velocity is read.

    Returns:
        The overshoot in ``[0, max_penalty]``. Shape is (num_envs,).
    """
    return (_ball_forward_speed(env, asset_cfg) - target_speed).clamp(0.0, max_penalty)


def single_foot_grounded_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward one named foot being on the ground, as a 0-or-1 flag.

    Args:
        env: The environment instance.
        sensor_cfg: The terrain-filtered contact sensor and the sensing objects to read.

    Returns:
        Whether the foot is on the ground, as 0.0 or 1.0. Shape is (num_envs,).
    """
    return _any_contact(env, sensor_cfg, filtered=True).float()


_POSTURE_KEYFRAME_ATTR = "_microduck_posture_keyframe_cache"

_POSTURE_RAMP_DONE_TOLERANCE = 0.02

"""Blend error below which :func:`posture_stillness` treats the posture ramp as finished."""


def _posture_blend(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The commanded posture blend: 0 at the stand target, 1 at the sit target.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.

    Returns:
        The blend in ``[0, 1]``. Shape is (num_envs,).

    Raises:
        ValueError: If the named command term exposes no ``alpha``.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is None:
        raise ValueError(
            f"The posture rewards read the slewed blend of the command term '{command_name}', which"
            " exposes no 'alpha'. Configure it as a"
            " 'isaaclab_tasks.contrib.microduck.mdp.SitStandCommandCfg'."
        )
    return alpha


def _keyframe_in_selection(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, keyframe: Mapping[str, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve, and cache on the environment, a named joint keyframe inside a joint selection.

    Shared by the sit/stand posture rewards and the roller-crouch pose rewards, which differ in where
    the blend comes from -- a slewed command flag in one case, the cycle phase in the other -- and not
    in how a keyframe is matched to a selection.

    Args:
        env: The environment instance, which carries the cache.
        asset_cfg: The resolved joint selection the keyframe is written into.
        keyframe: Joint name to angle [rad].

    Returns:
        The override positions and their angles [rad]. Both have shape (num_overrides,).

    Raises:
        ValueError: If the selection is not by name, or if it omits a keyframe joint.
    """
    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = env.__dict__.setdefault(_POSTURE_KEYFRAME_ATTR, {})
    key = (asset_cfg.name, tuple(asset_cfg.joint_names or ()), tuple(keyframe.items()))
    resolved = cache.get(key)
    if resolved is None:
        names = list(asset_cfg.joint_names or ())
        if not names:
            raise ValueError(
                "The keyframe rewards need their joints selected by name, so that the keyframe can be"
                " matched against them. Set 'joint_names' with 'preserve_order=True'."
            )
        missing = [name for name in keyframe if name not in names]
        if missing:
            raise ValueError(
                f"The keyframe sets {missing}, which the scored joint selection {names} does not"
                " contain, so those joints would be rewarded against the stand pose instead."
            )
        positions = torch.tensor([names.index(name) for name in keyframe], device=env.device, dtype=torch.long)
        angles = torch.tensor(list(keyframe.values()), device=env.device)
        resolved = (positions, angles)
        cache[key] = resolved
    return resolved


def _posture_joint_target(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sit_joint_pos: Mapping[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The selected joints' commanded target and their measured positions [rad].

    The target interpolates the stand pose toward the sitting keyframe by the commanded blend, so
    mid-ramp the rewarded pose folds in step with the descending height target.
    """
    positions, angles = _keyframe_in_selection(env, asset_cfg, sit_joint_pos)
    stand_target = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    sit_target = stand_target.clone()
    sit_target[:, positions] = angles
    blend = _posture_blend(env, command_name).unsqueeze(-1)
    target = stand_target + blend * (sit_target - stand_target)
    return asset.data.joint_pos.torch[:, asset_cfg.joint_ids], target


def _posture_height(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    command_name: str,
    sit_height: float,
    stand_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The commanded target trunk height and the measured one [m], both above the environment origin."""
    blend = _posture_blend(env, command_name)
    target_height = stand_height + blend * (sit_height - stand_height)
    return target_height, _root_height_above_ground(env, asset)


def posture_pose_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_joint_pos: Mapping[str, float],
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward holding selected joints at the *commanded* posture's pose, with a Gaussian tolerance.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        sit_joint_pos: Joint name to angle [rad] of the sitting keyframe. Joints left out of it are
            rewarded at the stand pose in both postures, which is how the neck and head are handled.
        std: Width of the per-joint Gaussian kernel [rad].
        asset_cfg: The articulation and the joints scored. Select them by name with
            ``preserve_order=True``; the selection must contain every keyframe joint.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, target = _posture_joint_target(env, asset, asset_cfg, command_name, sit_joint_pos)
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def posture_pose_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_joint_pos: Mapping[str, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the mean absolute deviation of selected joints from the commanded posture's pose.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        sit_joint_pos: Joint name to angle [rad] of the sitting keyframe.
        asset_cfg: The articulation and the joints scored, selected by name.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, target = _posture_joint_target(env, asset, asset_cfg, command_name, sit_joint_pos)
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_height: float,
    stand_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the trunk at the commanded posture's height, using a Gaussian kernel.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        sit_height: Trunk height [m] of the seated rest.
        stand_height: Trunk height [m] of the standing rest.
        std: Width of the Gaussian kernel [m].
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_height, height = _posture_height(env, asset, command_name, sit_height, stand_height)
    return torch.exp(-(((height - target_height) / std) ** 2))


def posture_height_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the absolute distance of the trunk from the commanded posture's height.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        sit_height: Trunk height [m] of the seated rest.
        stand_height: Trunk height [m] of the standing rest.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_height, height = _posture_height(env, asset, command_name, sit_height, stand_height)
    return -torch.abs(height - target_height)


def posture_rise_bootstrap(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_height: float,
    max_vz: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward rising while a *stand* is commanded, below a ceiling and up to a speed cap.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        max_height: Trunk height [m] above which the reward is switched off. Set just above the
            standing rest, so the final centimetre still pays.
        max_vz: Upward speed [m/s] the payout saturates at.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, max_vz]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sit_flag = env.command_manager.get_command(command_name)[:, 0]
    below_ceiling = (_root_height_above_ground(env, asset) < max_height).float()
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return torch.clamp(vertical_speed, min=0.0, max=max_vz) * below_ceiling * (1.0 - sit_flag)


def trunk_downward_velocity_penalty(
    env: ManagerBasedRLEnv,
    max_down_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize downward trunk speed beyond a cap.

    Args:
        env: The environment instance.
        max_down_vel: Downward speed [m/s] below which nothing is charged.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return -torch.clamp(-vertical_speed - max_down_vel, min=0.0)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRLEnv,
    max_up_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize upward trunk speed beyond a cap.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        max_up_vel: Upward speed [m/s] below which nothing is charged.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return -torch.clamp(vertical_speed - max_up_vel, min=0.0)


def upright_linear_at_height(
    env: ManagerBasedRLEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the cosine of the trunk's tilt, in proportion to how tall the robot is standing.

    Args:
        env: The environment instance.
        height_low: Trunk height [m] below which the gate is fully closed.
        height_high: Trunk height [m] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[-1, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    uprightness = 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)
    return uprightness * _height_gate(_root_height_above_ground(env, asset), height_low, height_high)


def posture_stillness(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_height: float,
    stand_height: float,
    band_full: float,
    band_zero: float,
    vel_std: float,
    tilt_full_deg: float,
    tilt_zero_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward resting still, upright and at the commanded height, once the transition has finished.

    * a **height band** around the commanded target, so the term is inactive mid-transition and
      cannot pay for stopping half-way;
    * a **tilt** smoothstep, so a back, face or side flop -- motionless, and inside the seated height
      band -- earns nothing;
    * **ramp completion**, ``|flag - blend| < 0.02``, so stillness never pays while the setpoint is
      still moving. This is what makes "arrive, then hold" the peak of the stack rather than "stay
      where you are".

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        sit_height: Trunk height [m] of the seated rest.
        stand_height: Trunk height [m] of the standing rest.
        band_full: Height error [m] below which the height gate is fully open.
        band_zero: Height error [m] above which the height gate is fully closed.
        vel_std: Width of the Gaussian kernel on the trunk speed [m/s].
        tilt_full_deg: Trunk tilt [deg] below which the tilt gate is fully open.
        tilt_zero_deg: Trunk tilt [deg] above which the tilt gate is fully closed.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_height, height = _posture_height(env, asset, command_name, sit_height, stand_height)
    speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    ramp_done = ((flag - _posture_blend(env, command_name)).abs() < _POSTURE_RAMP_DONE_TOLERANCE).float()

    height_gate = _smoothstep((band_zero - torch.abs(height - target_height)) / max(band_zero - band_full, 1e-6))

    tilt_gate = _cos_tilt_gate(asset.data.root_link_quat_w.torch, tilt_full_deg, tilt_zero_deg)
    return torch.exp(-((speed / vel_std) ** 2)) * height_gate * tilt_gate * ramp_done


def posture_composite(
    env: ManagerBasedRLEnv,
    command_name: str,
    sit_joint_pos: Mapping[str, float],
    sit_height: float,
    stand_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    head_std: float,
    head_command_name: str,
    asset_cfg: SceneEntityCfg,
    head_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the commanded posture as one multiplicative score over height, tilt, pose and head.

    The upright factor is posture-independent: both rest states demand a vertical trunk.

    Args:
        env: The environment instance.
        command_name: Name of the posture command term.
        sit_joint_pos: Joint name to angle [rad] of the sitting keyframe.
        sit_height: Trunk height [m] of the seated rest.
        stand_height: Trunk height [m] of the standing rest.
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        pose_std: Width of the Gaussian kernel on the joint-position RMS error [rad].
        head_std: Width of the Gaussian kernel on the head-tracking RMS error [rad].
        head_command_name: Name of the head-pose command term.
        asset_cfg: The articulation and the joints scored against the posture pose, selected by name.
            Its root link carries the trunk height and orientation.
        head_asset_cfg: The head joints scored against the head-pose command, selected by name with
            ``preserve_order=True`` so their columns match that command.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, target = _posture_joint_target(env, asset, asset_cfg, command_name, sit_joint_pos)
    target_height, height = _posture_height(env, asset, command_name, sit_height, stand_height)

    height_score = torch.exp(-(((height - target_height) / height_std) ** 2))
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / upright_std**2)
    pose_score = torch.exp(-torch.square(joint_pos - target).mean(dim=-1) / pose_std**2)

    head_asset: Articulation = env.scene[head_asset_cfg.name]
    head_command = env.command_manager.get_command(head_command_name)
    head_measured = head_asset.data.joint_pos.torch[:, head_asset_cfg.joint_ids]
    head_error = (head_measured - head_asset.data.default_joint_pos.torch[:, head_asset_cfg.joint_ids]) - head_command
    head_score = torch.exp(-torch.square(head_error).mean(dim=-1) / head_std**2)

    return height_score * upright_score * pose_score * head_score


def neck_torque_when_tilted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    tilt_start_deg: float = 12.0,
    tilt_full_deg: float = 25.0,
    torque_scale: float = 0.1,
) -> torch.Tensor:
    """Charge head-joint strain during backward lean in the slewed seated target.

    Args:
        env: The environment instance.
        asset_cfg: Articulation and selected neck/head joints.
        command_name: Name of the :class:`SitStandCommand` posture request.
        tilt_start_deg: Backward tilt [deg] below which the cost is zero.
        tilt_full_deg: Backward tilt [deg] above which the tilt gate is fully open.
        torque_scale: Torque [N.m] used to normalize the mean squared head-joint load.

    Returns:
        The nonnegative gated strain cost. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.nan_to_num(asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids], nan=0.0)
    cost = torch.square(torque / torque_scale).mean(dim=1)
    gravity = torch.nan_to_num(asset.data.projected_gravity_b.torch, nan=0.0)
    tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
    gate = torch.clamp((tilt - tilt_start_deg) / max(tilt_full_deg - tilt_start_deg, 1e-6), 0.0, 1.0)
    return cost * gate * (gravity[:, 0] < 0.0).float() * _posture_blend(env, command_name)


"""The two gates are **not** complements. ``_phase_pose_blend`` ramps 0 to 1 across the descent, holds
at 1 through the low dwell and falls back to 0 across the rise; ``_phase_rise_gate`` is 0 until the
low dwell ends, ramps to 1 across the rise and stays there through the standing rest. They sum to 1
across the rise and nowhere else -- during the descent the down-gate is opening while the up-gate is
still shut, which is what leaves the approach unpriced by the return terms.
"""


def _phase_from_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Recover the cycle phase in ``[0, 1)`` from a ``(cos, sin, 0)`` command slot.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.

    Returns:
        The phase in ``[0, 1)``. Shape is (num_envs,).
    """
    command = env.command_manager.get_command(command_name)
    return (torch.atan2(command[:, 1], command[:, 0]) / (2.0 * math.pi)) % 1.0


def _phase_pose_blend(phase: torch.Tensor, descent_end: float, hold_end: float, rise_end: float) -> torch.Tensor:
    """Gate that follows the bend: 0 standing, ramping to 1 by ``descent_end``, back to 0 by ``rise_end``.

    Args:
        phase: Position in the cycle, in ``[0, 1)``. Shape is (num_envs,).
        descent_end: Phase at which the descent finishes and the low dwell begins.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.

    Returns:
        The gate in ``[0, 1]``. Shape is (num_envs,).
    """
    blend = torch.zeros_like(phase)
    blend = torch.where(phase < descent_end, phase / max(descent_end, 1e-6), blend)
    blend = torch.where((phase >= descent_end) & (phase < hold_end), torch.ones_like(phase), blend)
    rising = (phase >= hold_end) & (phase < rise_end)
    blend = torch.where(rising, 1.0 - (phase - hold_end) / max(rise_end - hold_end, 1e-6), blend)
    return blend


def _phase_rise_gate(phase: torch.Tensor, hold_end: float, rise_end: float) -> torch.Tensor:
    """Gate that follows the return: 0 before ``hold_end``, ramping to 1 by ``rise_end``, 1 after.

    Args:
        phase: Position in the cycle, in ``[0, 1)``. Shape is (num_envs,).
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.

    Returns:
        The gate in ``[0, 1]``. Shape is (num_envs,).
    """
    gate = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    gate = torch.where(rising, (phase - hold_end) / max(rise_end - hold_end, 1e-6), gate)
    return torch.where(phase >= rise_end, torch.ones_like(phase), gate)


def _mouth_tip_pose_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, offset_b: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """World position [m] of the mouth tip and the orientation of the body carrying it.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the single body the mouth tip is rigidly attached to.
        offset_b: Mouth-tip position [m] in that body's frame.

    Returns:
        The mouth-tip world position [m], shape (num_envs, 3), and the carrying body's orientation in
        (x, y, z, w), shape (num_envs, 4).

    Raises:
        ValueError: If ``asset_cfg`` does not select exactly one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None or len(asset_cfg.body_ids) != 1:
        raise ValueError(
            "The mouth terms measure one body the mouth tip is attached to; 'asset_cfg' must select"
            f" exactly one by name. Received: {asset_cfg.body_names}."
        )
    body_pos_w = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids].squeeze(1)
    body_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids].squeeze(1)
    offset = torch.tensor(tuple(offset_b), dtype=body_pos_w.dtype, device=body_pos_w.device)
    return body_pos_w + math_utils.quat_apply(body_quat_w, offset.expand_as(body_pos_w)), body_quat_w


def mouth_ground_proximity_phased(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mouth_offset_b: Sequence[float],
    std: float,
    command_name: str,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    target_height: float = 0.0,
) -> torch.Tensor:
    """Reward the mouth tip being close to the ground, while the cycle is asking for the bend.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the single body the mouth tip is attached to.
        mouth_offset_b: Mouth-tip position [m] in that body's frame.
        std: Width of the Gaussian kernel on the mouth height [m].
        command_name: Name of the phase command term.
        descent_end: Phase at which the descent finishes and the low dwell begins.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.
        target_height: Mouth height [m] the kernel peaks at. Defaults to 0.0, the floor.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    mouth_pos_w, _ = _mouth_tip_pose_w(env, asset_cfg, mouth_offset_b)
    height = mouth_pos_w[:, 2] - env.scene.env_origins[:, 2]
    proximity = torch.exp(-(((height - target_height) / std) ** 2))
    gate = _phase_pose_blend(_phase_from_command(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mouth_axis_b: Sequence[float],
    command_name: str,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Reward the mouth axis pointing at the floor, while the cycle is asking for the bend.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the single body the mouth tip is attached to.
        mouth_axis_b: The mouth's pointing axis [-] in that body's frame.
        command_name: Name of the phase command term.
        descent_end: Phase at which the descent finishes and the low dwell begins.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.

    Returns:
        The reward in ``[-1, 1]``. Shape is (num_envs,).
    """
    _, body_quat_w = _mouth_tip_pose_w(env, asset_cfg, (0.0, 0.0, 0.0))
    axis = torch.tensor(tuple(mouth_axis_b), dtype=body_quat_w.dtype, device=body_quat_w.device)
    axis = torch.nn.functional.normalize(axis, dim=-1).expand(body_quat_w.shape[0], 3)
    alignment = -math_utils.quat_apply(body_quat_w, axis)[:, 2]
    gate = _phase_pose_blend(_phase_from_command(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    command_name: str,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Reward selected joints returning to the stand pose, while the cycle is asking for the return.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the joints to return to the stand pose.
        std: Width of the per-joint Gaussian kernel [rad].
        command_name: Name of the phase command term.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    gate = _phase_rise_gate(_phase_from_command(env, command_name), hold_end, rise_end)
    return gate * joint_pose_gaussian(env, std, asset_cfg)


def ground_pick_return_upright_phased(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    hold_end: float,
    rise_end: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward an upright trunk, while the cycle is asking for the return.

    Its kernel is ``1 - cos(tilt)`` where :func:`upright_selected_body` uses the projected gravity direction, so
    the two are not the same width at the same ``std`` and cannot be merged.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        command_name: Name of the phase command term.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / std**2)
    gate = _phase_rise_gate(_phase_from_command(env, command_name), hold_end, rise_end)
    return gate * upright_score


def neck_vel_descent_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    hold_end: float,
) -> torch.Tensor:
    """Penalize neck joint speed while the cycle is asking for the bend and the low dwell.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the neck joints to charge.
        command_name: Name of the phase command term.
        hold_end: Phase at which the low dwell finishes and the rise begins.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cost = torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]).mean(dim=-1)
    gate = (_phase_from_command(env, command_name) < hold_end).to(cost.dtype)
    return gate * cost


def feet_grounded_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward the fraction of the selected feet that are touching the ground.

    Args:
        env: The environment instance.
        sensor_cfg: The terrain-filtered contact sensor and the sole colliders to read.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"The contact sensor '{sensor_cfg.name}' reports no force matrix. Set"
            " 'filter_prim_paths_expr' or 'filter_shape_prim_expr' on its configuration so that its"
            " contact partners are resolved."
        )
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    grounded = (forces.norm(dim=-1) > 0.0).any(dim=-1)
    return grounded.float().mean(dim=-1)


def body_impact_cost(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Charge the ground-contact force on selected colliders, above a threshold.

    The stock :func:`isaaclab.envs.mdp.undesired_contacts` counts contacts over a force threshold
    instead of charging the excess, so it has no gradient to descend once a contact exists.

    Args:
        env: The environment instance.
        sensor_cfg: The terrain-filtered contact sensor and the colliders to charge.
        threshold: Contact-force magnitude [N] below which nothing is charged.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"The contact sensor '{sensor_cfg.name}' reports no force matrix. Set"
            " 'filter_prim_paths_expr' or 'filter_shape_prim_expr' on its configuration so that its"
            " contact partners are resolved."
        )
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    total_force = forces.flatten(start_dim=1, end_dim=-2).sum(dim=1)
    return torch.clamp(total_force.norm(dim=-1) - threshold, min=0.0)


def _crouch_pose_target(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    crouch_pose: Mapping[str, float],
    stand_pose: Mapping[str, float] | None,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The selected joints' phase-blended pose target and their measured positions [rad].

    The source pose is the articulation's stand pose with :attr:`stand_pose` written over it, the
    destination is :attr:`crouch_pose`, and the blend is the cycle's descent/hold/rise envelope.
    """
    joint_ids = asset_cfg.joint_ids
    source = asset.data.default_joint_pos.torch[:, joint_ids].clone()
    if stand_pose:
        positions, angles = _keyframe_in_selection(env, asset_cfg, stand_pose)
        source[:, positions] = angles
    target = source.clone()
    positions, angles = _keyframe_in_selection(env, asset_cfg, crouch_pose)
    target[:, positions] = angles

    blend = _phase_pose_blend(_phase_from_command(env, command_name), descent_end, hold_end, rise_end)
    blended = source + blend.unsqueeze(-1) * (target - source)
    return asset.data.joint_pos.torch[:, joint_ids], blended


def crouch_glide_pose_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    crouch_pose: Mapping[str, float],
    stand_pose: Mapping[str, float],
    std: float,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward holding the selected joints at the phase's blended crouch pose, with a Gaussian tolerance.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        crouch_pose: Joint name to angle [rad] of the crouched keyframe.
        stand_pose: Joint name to angle [rad] of the standing keyframe the cycle departs from and
            returns to. Joints left out of it stay at the articulation's stand pose.
        std: Width of the per-joint Gaussian kernel [rad].
        descent_end: Phase at which the descent finishes and the low dwell begins.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.
        asset_cfg: The articulation and the joints scored. Select them by name with
            ``preserve_order=True``; the selection must contain every keyframe joint.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, target = _crouch_pose_target(
        env, asset, asset_cfg, command_name, crouch_pose, stand_pose, descent_end, hold_end, rise_end
    )
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def crouch_glide_pose_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    crouch_pose: Mapping[str, float],
    stand_pose: Mapping[str, float],
    descent_end: float,
    hold_end: float,
    rise_end: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the mean absolute deviation of the selected joints from the phase's blended crouch pose.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        crouch_pose: Joint name to angle [rad] of the crouched keyframe.
        stand_pose: Joint name to angle [rad] of the standing keyframe.
        descent_end: Phase at which the descent finishes and the low dwell begins.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.
        asset_cfg: The articulation and the joints scored, selected by name.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, target = _crouch_pose_target(
        env, asset, asset_cfg, command_name, crouch_pose, stand_pose, descent_end, hold_end, rise_end
    )
    return -torch.abs(joint_pos - target).mean(dim=-1)


def forward_speed_reward(
    env: ManagerBasedRLEnv, vel_ref: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward rolling forward, saturating at a reference speed.

    The stock :func:`isaaclab.envs.mdp.track_lin_vel_xy_exp` tracks a *commanded* velocity and would
    charge going faster than asked; this one only ever pays, and only forwards.

    Args:
        env: The environment instance.
        vel_ref: Forward speed [m/s] the ``tanh`` is scaled to saturate near.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forward_vel = asset.data.root_link_lin_vel_b.torch[:, 0]
    return torch.tanh(torch.clamp(forward_vel, min=0.0) / vel_ref)


def crouch_forward_lean(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pitch: float,
    std: float,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a slight forward trunk lean, only while the cycle is asking for the crouch.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        target_pitch: Sine of the requested nose-down lean, in ``[-1, 1]``.
        std: Width [-] of the Gaussian on the lean error.
        descent_end: Phase at which the descent finishes and the low dwell begins.
        hold_end: Phase at which the low dwell finishes and the rise begins.
        rise_end: Phase at which the rise finishes and the standing rest begins.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gate = _phase_pose_blend(_phase_from_command(env, command_name), descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b.torch[:, 0]
    return gate * torch.exp(-torch.square(lean - target_pitch) / std**2)


"""
The spin trick asks for a target yaw *rate* rather than a pose, so its phase envelope is a trapezoid
in angular velocity -- launch, hold, brake, rest -- instead of the crouch family's pose blend. The
same clock drives both: the phase comes out of the twist slot the same way, and the envelope
normalized to ``[0, 1]`` is what gates the two shaping terms, so the leg scissor and the wheel
differential fade out before the standing rest and the robot hands back in a neutral station.
"""


def _spin_rate_by_phase(
    phase: torch.Tensor, rate_max: float, accel_end: float, hold_end: float, brake_end: float
) -> torch.Tensor:
    """The commanded yaw rate [rad/s] along the cycle, positive counter-clockwise.

    A trapezoid: a linear ramp to :attr:`rate_max` by :attr:`accel_end`, a hold to :attr:`hold_end`,
    a linear ramp back to zero by :attr:`brake_end`, then zero for the standing rest.

    Args:
        phase: Position in the cycle, in ``[0, 1)``. Shape is (num_envs,).
        rate_max: Peak yaw rate [rad/s] of the hold segment.
        accel_end: Phase at which the launch finishes and the hold begins.
        hold_end: Phase at which the hold finishes and the braking begins.
        brake_end: Phase at which the braking finishes and the standing rest begins.

    Returns:
        The target yaw rate [rad/s] in ``[0, rate_max]``. Shape is (num_envs,).
    """
    rate = torch.zeros_like(phase)
    rate = torch.where(phase < accel_end, rate_max * phase / max(accel_end, 1e-6), rate)
    rate = torch.where((phase >= accel_end) & (phase < hold_end), torch.full_like(phase, rate_max), rate)
    braking = (phase >= hold_end) & (phase < brake_end)
    rate = torch.where(braking, rate_max * (1.0 - (phase - hold_end) / max(brake_end - hold_end, 1e-6)), rate)
    return rate


def _spin_target_rate(
    env: ManagerBasedRLEnv, command_name: str, rate_max: float, accel_end: float, hold_end: float, brake_end: float
) -> torch.Tensor:
    """The commanded yaw rate [rad/s] this step, read off the phase command."""
    return _spin_rate_by_phase(_phase_from_command(env, command_name), rate_max, accel_end, hold_end, brake_end)


def _spin_gate(
    env: ManagerBasedRLEnv, command_name: str, rate_max: float, accel_end: float, hold_end: float, brake_end: float
) -> torch.Tensor:
    """The envelope normalized to ``[0, 1]``, which is what the two shaping terms are gated on.

    It is zero across the whole standing rest and *not* zero during the braking ramp, so the shaping
    fades out with the rotation instead of being cut at the top of it.
    """
    return _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_rate_track(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the cycle's commanded yaw rate, with a Gaussian tolerance.

    The measured rate is taken in the **body** frame, deliberately: that is what the robot's own gyro
    reports and therefore what the policy observes.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        std: Width [rad/s] of the Gaussian on the yaw-rate error.
        rate_max: Peak yaw rate [rad/s] of the envelope's hold segment.
        accel_end: Phase at which the launch finishes and the hold begins.
        hold_end: Phase at which the hold finishes and the braking begins.
        brake_end: Phase at which the braking finishes and the standing rest begins.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate = asset.data.root_link_ang_vel_b.torch[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return torch.exp(-torch.square((yaw_rate - target) / std))


def spin_rate_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the absolute yaw-rate error against the cycle's commanded rate.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        rate_max: Peak yaw rate [rad/s] of the envelope's hold segment.
        accel_end: Phase at which the launch finishes and the hold begins.
        hold_end: Phase at which the hold finishes and the braking begins.
        brake_end: Phase at which the braking finishes and the standing rest begins.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate = asset.data.root_link_ang_vel_b.torch[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(yaw_rate - target)


def spin_stay_in_place(
    env: ManagerBasedRLEnv,
    command_name: str,
    launch_scale: float,
    accel_end: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Charge the trunk's horizontal speed, discounted while the rotation is being launched.

    Two details are deliberate. The launch discount lets the policy shuffle its feet to get the
    rotation started, and the term is **not** gated by the envelope: during the standing rest the
    charge is at full strength, because that is exactly when the robot is supposed to be still.

    The term returns a **non-negative** cost and is therefore configured with a *negative* weight.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        launch_scale: Fraction of the cost charged before :attr:`accel_end`, in ``[0, 1]``.
        accel_end: Phase at which the launch finishes and the full charge begins.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cost = torch.sum(torch.square(asset.data.root_link_lin_vel_b.torch[:, :2]), dim=1)
    phase = _phase_from_command(env, command_name)
    scale = torch.where(phase < accel_end, torch.full_like(cost, launch_scale), torch.ones_like(cost))
    return cost * scale


def spin_wheel_differential(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_wheel_cfg: SceneEntityCfg,
    right_wheel_cfg: SceneEntityCfg,
    omega_scale: float,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    """Reward the wheels turning in opposite directions, in the sense that spins counter-clockwise.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        left_wheel_cfg: The articulation and the left bogie's wheel hinges, averaged.
        right_wheel_cfg: The articulation and the right bogie's wheel hinges, averaged.
        omega_scale: Wheel-rate difference [rad/s] the ``tanh`` is scaled to saturate near.
        rate_max: Peak yaw rate [rad/s] of the envelope's hold segment.
        accel_end: Phase at which the launch finishes and the hold begins.
        hold_end: Phase at which the hold finishes and the braking begins.
        brake_end: Phase at which the braking finishes and the standing rest begins.

    Returns:
        The reward in ``[0, 1)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[left_wheel_cfg.name]
    joint_vel = asset.data.joint_vel.torch
    left_rate = joint_vel[:, left_wheel_cfg.joint_ids].mean(dim=1)
    right_rate = joint_vel[:, right_wheel_cfg.joint_ids].mean(dim=1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * torch.tanh(torch.clamp(right_rate - left_rate, min=0.0) / omega_scale)


def spin_grounded(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
    bodies_per_foot: int = 1,
) -> torch.Tensor:
    """Reward keeping both blades on the ground, while the cycle is asking for rotation.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read, in per-foot order.
        command_name: Name of the phase command term.
        rate_max: Peak yaw rate [rad/s] of the envelope's hold segment.
        accel_end: Phase at which the launch finishes and the hold begins.
        hold_end: Phase at which the hold finishes and the braking begins.
        brake_end: Phase at which the braking finishes and the standing rest begins.
        bodies_per_foot: Contact bodies making up one foot. Defaults to 1.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    grounded = (_feet_in_contact(env, sensor_cfg, bodies_per_foot) >= 2.0).float()
    return grounded * _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)


def leg_antisymmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_joint_cfg: SceneEntityCfg,
    right_joint_cfg: SceneEntityCfg,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    """Reward a scissored leg pose while the cycle is asking for rotation.

    It is a training wheel rather than an objective, and the task decays its weight by curriculum so
    the policy is left to refine its own pumping frequency once the mechanism has been found. The
    envelope gate is what stops it shaping the standing rest.

    Args:
        env: The environment instance.
        command_name: Name of the phase command term.
        left_joint_cfg: The articulation and the left-leg joints, in the order they pair up.
        right_joint_cfg: The articulation and the right-leg joints, in the matching order.
        rate_max: Peak yaw rate [rad/s] of the envelope's hold segment.
        accel_end: Phase at which the launch finishes and the hold begins.
        hold_end: Phase at which the hold finishes and the braking begins.
        brake_end: Phase at which the braking finishes and the standing rest begins.

    Returns:
        The reward in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[left_joint_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    scissor = -torch.abs(joint_pos[:, left_joint_cfg.joint_ids] - joint_pos[:, right_joint_cfg.joint_ids]).mean(dim=-1)
    return _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end) * scissor
