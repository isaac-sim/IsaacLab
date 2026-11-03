# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck encoder, IMU, foot-contact and task-state observations."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_from_angle_axis
from isaaclab.utils.noise import NoiseCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import quat_apply_inverse

from .events import encoder_bias

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor

"""
Sensor-model observations.
"""


def joint_pos_rel_biased(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    biased: bool = True,
    action_name: str = "joint_pos",
    backlash_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Encoder offsets from the default pose [rad], optionally including paired gearbox play.

    ``asset_cfg`` must select the joints of the ``action_name`` action term, in the same order, and
    ``backlash_cfg`` the passive hinges in that order. Calibration bias belongs to the servo encoder only.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if biased:
        joint_pos = joint_pos + env.action_manager.get_term(action_name).encoder_bias
    if backlash_cfg is not None:
        joint_pos = joint_pos + env.scene[backlash_cfg.name].data.joint_pos.torch[:, backlash_cfg.joint_ids]
    return joint_pos - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


def joint_vel_rel_backlash(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, backlash_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Encoder velocity relative to the default [rad/s], including the paired play hinges.

    Both selections must list corresponding servo and passive joints in the same order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    velocity = velocity + env.scene[backlash_cfg.name].data.joint_vel.torch[:, backlash_cfg.joint_ids]
    return velocity - asset.data.default_joint_vel.torch[:, asset_cfg.joint_ids]


class misaligned_imu(ManagerTermBase):
    """Root angular velocity [rad/s] then projected gravity, read by a misaligned IMU with bus latency."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        axis = torch.nn.functional.normalize(torch.randn(self.num_envs, 3, device=self.device), dim=-1)
        angle = torch.rand(self.num_envs, device=self.device) * math.radians(cfg.params["max_angle_deg"])
        self._quat = quat_from_angle_axis(angle, axis)
        self._min_lag = cfg.params["min_lag"]
        self._max_lag = cfg.params["max_lag"]
        self._update_period = cfg.params["update_period"]
        self._buffer = CircularBuffer(max_len=self._max_lag + 1, batch_size=self.num_envs, device=self.device)
        self._lags = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_offsets = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self._pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_pending_reset = False
        self._last_step: int | None = None
        self._output: torch.Tensor | None = None
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        resolved = slice(None) if env_ids is None else env_ids
        self._lags[resolved] = 0
        self._step_count[resolved] = 0
        phases = torch.randint(0, self._update_period, (self.num_envs, 2), device=self.device)
        self._phase_offsets[resolved] = phases[resolved]
        self._pending_reset[resolved] = True
        self._has_pending_reset = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_angle_deg: float,
        min_lag: int,
        max_lag: int,
        update_period: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Return each environment's reading from its current lag, shape (num_envs, 6)."""
        # Repeated reads within one step return the same sample, except for freshly reset environments.
        repeated_read = env.common_step_counter == self._last_step
        if repeated_read and not self._has_pending_reset:
            return self._output
        asset: Articulation = env.scene[asset_cfg.name]
        vectors = torch.stack((asset.data.root_ang_vel_b.torch, asset.data.projected_gravity_b.torch), dim=1)
        reading = quat_apply(self._quat.unsqueeze(1), vectors).flatten(start_dim=1)
        if repeated_read:
            return torch.where(self._pending_reset.unsqueeze(-1), reading, self._output)
        if self._has_pending_reset:
            self._buffer.reset(self._pending_reset)
            self._pending_reset.fill_(False)
            self._has_pending_reset = False
        self._last_step = env.common_step_counter
        self._buffer.append(reading)
        should_update = (self._step_count.unsqueeze(-1) + self._phase_offsets) % self._update_period == 0
        candidates = torch.randint(self._min_lag, self._max_lag + 1, (self.num_envs, 2), device=self.device)
        self._lags = torch.where(should_update, candidates, self._lags)
        self._step_count += 1
        self._output = torch.cat((self._buffer[self._lags[:, 0]][:, :3], self._buffer[self._lags[:, 1]][:, 3:]), dim=-1)
        return self._output


def imu_noise(data: torch.Tensor, cfg: ImuNoiseCfg) -> torch.Tensor:
    """Add uniform noise within ``cfg.ang_vel`` to angular velocity and within ``cfg.gravity`` to gravity."""
    noise = torch.rand_like(data) * 2.0 - 1.0
    return data + torch.cat((noise[:, :3] * cfg.ang_vel, noise[:, 3:] * cfg.gravity), dim=-1)


@configclass
class ImuNoiseCfg(NoiseCfg):
    """Additive uniform noise on :class:`misaligned_imu`, with one bound per reading."""

    func: Callable[[torch.Tensor, ImuNoiseCfg], torch.Tensor] = imu_noise
    ang_vel: float = MISSING
    """Angular-velocity noise bound [rad/s]."""
    gravity: float = MISSING
    """Projected-gravity noise bound."""


"""
Privileged foot observations.
"""


def zero_non_finite(value: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries with zero."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Whether each selected foot is in contact with anything."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    in_contact = (forces.norm(dim=-1) > 0.0).float()
    return in_contact


def foot_contact_forces_safe(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact force on each selected body, log-compressed and guarded against NaNs."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    forces = forces.flatten(start_dim=1)
    return zero_non_finite(torch.sign(forces) * torch.log1p(forces.abs()))


def foot_air_time_safe(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Air time [s] per foot, with non-finite values replaced by zero."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return zero_non_finite(sensor.data.current_air_time.torch[:, sensor_cfg.body_ids])


def foot_height(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, height_sensor_names: tuple[str, ...] = ()
) -> torch.Tensor:
    """Foot-frame clearance [m], using the closest terrain ray per foot when supplied.

    Sensors must follow the selected bodies' order. Missed rays are ignored; when all rays
    miss, clearance falls back to the terrain origin, bounded by the sensor range.
    Without sensors, the terrain origin defines the flat ground height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    flat_heights = heights - env.scene.env_origins[:, 2].unsqueeze(-1)
    if not height_sensor_names:
        return flat_heights
    clearances = []
    for index, name in enumerate(height_sensor_names):
        sensor = env.scene[name]
        data = sensor.data
        hit_z = data.ray_hits_w.torch[..., 2]
        valid = torch.isfinite(hit_z)
        clearance = (heights[:, index, None] - hit_z).clamp(min=0.0)
        # A ray inside terrain can hit the underside of a step.
        clearance = torch.where(data.ray_normals_w.torch[..., 2] < 0.0, 0.0, clearance)
        clearance = torch.where(valid, clearance, sensor.cfg.max_distance).amin(dim=-1)
        fallback = flat_heights[:, index].clamp(0.0, sensor.cfg.max_distance)
        clearances.append(torch.where(valid.any(dim=-1), clearance, fallback))
    return torch.stack(clearances, dim=-1)


def foot_height_safe(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, height_sensor_names: tuple[str, ...] = ()
) -> torch.Tensor:
    """Foot clearance [m] above terrain, with non-finite values replaced by zero."""
    return zero_non_finite(foot_height(env, asset_cfg, height_sensor_names))


_IMU_MISALIGNMENT_ATTR = "_microduck_imu_misalignment"

"""Attribute the per-environment IMU misalignment rotation is cached under on the environment."""

"""
Encoder bias.
"""


def joint_pos_rel_articulation_biased(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    biased: bool = True,
) -> torch.Tensor:
    """Joint positions w.r.t. the default joint positions, as the encoders report them.

    The bias is a *closed loop* with
    :class:`~isaaclab_tasks.contrib.microduck.mdp.actions.ArticulationEncoderBiasAction`, which
    subtracts the same offset from its position target: a policy commanding ``a`` reads ``a`` back
    while the true joint settles at ``default + a - bias``. What the bias perturbs is therefore the
    robot's actual posture, not the policy's frame of reference -- exactly the calibration error it
    models. Wiring this term without the matching action term breaks that loop and instead trains a
    permanent offset into the policy's own commands.

    Args:
        env: The environment.
        asset_cfg: The articulation and the joints to report, in the requested order.
        biased: Whether to add the encoder bias. Defaults to True.

    Returns:
        Joint positions relative to their defaults [m or rad, depending on joint type].
        Shape is (num_envs, num_selected_joints).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if biased:
        joint_pos = joint_pos + encoder_bias(env, asset_cfg)[:, asset_cfg.joint_ids]
    return joint_pos - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


"""
IMU misalignment.
"""


def _imu_misalignment_quat(env: ManagerBasedEnv, max_angle_rad: float) -> torch.Tensor:
    """The per-environment IMU mounting-misalignment rotation, sampled once per run.

    Returns:
        The misalignment quaternion in (x, y, z, w). Shape is (num_envs, 4).
    """
    cached: tuple[float, torch.Tensor] | None = getattr(env, _IMU_MISALIGNMENT_ATTR, None)
    if cached is not None:
        angle, quat = cached
        if angle != max_angle_rad:
            raise ValueError(
                "The IMU misalignment is one rotation shared by every misaligned observation term, so"
                f" they must agree on 'max_angle_deg'. Sampled with {math.degrees(angle)} deg, now asked"
                f" for {math.degrees(max_angle_rad)} deg."
            )
        return quat
    # a uniformly random axis, so the misalignment has no preferred direction
    axis = torch.randn(env.num_envs, 3, device=env.device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    angles = torch.rand(env.num_envs, device=env.device) * max_angle_rad
    quat = quat_from_angle_axis(angles, axis)
    setattr(env, _IMU_MISALIGNMENT_ATTR, (max_angle_rad, quat))
    return quat


def projected_gravity_imu_misaligned(
    env: ManagerBasedEnv,
    max_angle_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gravity projected on the root frame, seen through a misaligned IMU.

    The IMU on the real robot is bolted on with a small, unknown, and *fixed* orientation error, so
    the misalignment is drawn once per environment and never resampled -- it is a property of the
    robot, like a mass offset, not a per-step disturbance.

    Args:
        env: The environment.
        max_angle_deg: Upper bound [deg] on the misalignment angle. Zero disables the term, which
            then reproduces the stock observation exactly.
        asset_cfg: The articulation carrying the IMU.

    Returns:
        The misaligned gravity direction in the root frame. Shape is (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(quat, asset.data.projected_gravity_b.torch)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedEnv,
    max_angle_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root angular velocity, seen through the same misaligned IMU as the gravity term.

    See :func:`projected_gravity_imu_misaligned` for the misalignment model.

    Args:
        env: The environment.
        max_angle_deg: Upper bound [deg] on the misalignment angle.
        asset_cfg: The articulation carrying the IMU.

    Returns:
        The misaligned angular velocity [rad/s] in the root frame. Shape is (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(quat, asset.data.root_ang_vel_b.torch)


"""
Observation delay.
"""


class delayed_observation(ManagerTermBase):
    """Wraps another observation term in the stochastic bus latency models.

    * ``base_ang_vel`` and ``projected_gravity``: ``min_lag=0``, ``max_lag=1``,
      ``update_period=64`` -- a 0 or 20 ms latency that switches at most every 64 control steps;
    * ``joint_vel``: ``min_lag=1``, ``max_lag=1``, ``update_period=0`` -- a constant 20 ms lag,
      because the servo firmware derives velocity from the previous position-sample window.

    Rule 5 is why a reset does not clear the history immediately. An environment that computes a
    final observation before resetting part of the batch calls the term twice within one control
    step with a reset in between; clearing the history there and refilling it on the second call
    would advance the buffer twice, and every environment that was *not* reset would silently skip
    a frame. The clear is therefore deferred to the next advance, and the second call serves the
    reset environments the fresh value -- which is what they would read anyway, having no history
    left to be delayed by -- while the rest keep the frame they were already served.

    Isaac Lab's :class:`~isaaclab.utils.buffers.DelayBuffer` resamples its lag only on reset and
    has no ``update_period``, phase or hold concept, so this term keeps its own lag state over the
    same :class:`~isaaclab.utils.buffers.CircularBuffer` that ``DelayBuffer`` wraps. Doing the lag
    bookkeeping here also keeps it free of the host-device synchronization
    :meth:`~isaaclab.utils.buffers.DelayBuffer.set_time_lag` performs on every call.

    Never hoist an :class:`~isaaclab.managers.ObservationTermCfg` wrapping this term to module
    level and reuse the same instance across two observation slots (for example the actor's
    ``base_ang_vel`` and ``projected_gravity``): the manager keys its term instance -- and this
    term's lag state and :class:`~isaaclab.utils.buffers.CircularBuffer` -- off the cfg object's
    identity, so sharing one cfg instance aliases the delay buffer between the two slots instead
    of giving each its own. Construct a distinct cfg per slot.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._min_lag = int(cfg.params.get("min_lag", 0))
        self._max_lag = int(cfg.params.get("max_lag", 0))
        self._update_period = int(cfg.params.get("update_period", 0))
        self._hold_prob = float(cfg.params.get("hold_prob", 0.0))
        if self._max_lag < 1:
            raise ValueError(f"A delayed observation needs 'max_lag' >= 1, got {self._max_lag}.")
        if not 0 <= self._min_lag <= self._max_lag:
            raise ValueError(f"Expected 0 <= min_lag <= max_lag, got ({self._min_lag}, {self._max_lag}).")
        if self._update_period < 0:
            raise ValueError(f"A delayed observation needs 'update_period' >= 0, got {self._update_period}.")
        if not 0.0 <= self._hold_prob <= 1.0:
            raise ValueError(f"Expected 'hold_prob' in [0, 1], got {self._hold_prob}.")

        self._buffer = CircularBuffer(max_len=self._max_lag + 1, batch_size=self.num_envs, device=self.device)
        self._lags = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_offsets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # environments reset since the last advance; their history is cleared on the next one
        self._pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_pending_reset = False
        # the step the buffer was last advanced on, so a recomputed group does not advance it twice
        self._last_step: int | None = None
        self._output: torch.Tensor | None = None
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        resolved = slice(None) if env_ids is None else env_ids
        self._lags[resolved] = 0
        self._step_count[resolved] = 0
        if self._update_period > 0:
            phases = torch.randint(0, self._update_period, (self.num_envs,), device=self.device)
            self._phase_offsets[resolved] = phases[resolved]
        # the history itself is cleared on the next advance, so that a reset landing between two
        # computes of one control step cannot cost the untouched environments a frame
        self._pending_reset[resolved] = True
        self._has_pending_reset = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        term_func: Callable[..., torch.Tensor],
        term_params: dict | None = None,
        min_lag: int = 0,
        max_lag: int = 0,
        update_period: int = 0,
        hold_prob: float = 0.0,
    ) -> torch.Tensor:
        """Compute the wrapped term and return a stale copy of it.

        Args:
            env: The environment.
            term_func: The observation term to delay.
            term_params: The parameters to call ``term_func`` with. Defaults to None, which calls it
                with its own defaults.
            min_lag: Smallest lag [control steps] to draw. Defaults to 0.
            max_lag: Largest lag [control steps] to draw, inclusive. Defaults to 0, which is invalid
                and only present so the manager can validate the term signature.
            update_period: Control steps between two lag draws. Defaults to 0, which re-draws every
                step.
            hold_prob: Probability of keeping the previous lag on a draw step. Defaults to 0.0.

        Returns:
            The delayed observation, shaped like the wrapped term's output.
        """
        step = getattr(env, "common_step_counter", None)
        if step is not None and step == self._last_step and self._output is not None:
            if not self._has_pending_reset:
                return self._output
            # an environment reset since the buffer advanced has no history left to be delayed by,
            # so it reads the fresh value while the others keep the frame they were already served
            fresh = term_func(env, **(term_params or {}))
            mask = self._pending_reset.view(-1, *([1] * (fresh.dim() - 1)))
            return torch.where(mask, fresh, self._output)

        obs = term_func(env, **(term_params or {}))
        if self._has_pending_reset:
            self._buffer.reset(self._pending_reset)
            self._pending_reset.fill_(False)
            self._has_pending_reset = False
        self._last_step = step

        self._buffer.append(obs)
        self._update_lags()
        self._output = self._buffer[self._lags]
        return self._output

    def _update_lags(self) -> None:
        """Draw new lags for the environments whose update period has come round."""
        if self._update_period > 0:
            should_update = (self._step_count + self._phase_offsets) % self._update_period == 0
        else:
            should_update = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self._hold_prob > 0.0:
            should_update &= torch.rand(self.num_envs, device=self.device) >= self._hold_prob
        candidates = torch.randint(
            self._min_lag, self._max_lag + 1, (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._lags = torch.where(should_update, candidates, self._lags)
        self._step_count += 1


def zero_command_padding(env: ManagerBasedEnv, dim: int) -> torch.Tensor:
    """A block of zeros, standing in for a command the task does not have.

    Args:
        env: The environment instance.
        dim: Width of the slot to pad.

    Returns:
        Zeros. Shape is (num_envs, dim).
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


"""
NaN-safe critic terms.
"""


def fold_bodies_into_feet(values: torch.Tensor, bodies_per_foot: int) -> torch.Tensor:
    """Group a per-contact-body quantity into consecutive runs of bodies, one run per foot.

    Args:
        values: A per-body quantity in sensor-slot order. Shape is (num_envs, num_bodies,...).
        bodies_per_foot: Contact bodies making up one foot. ``1`` leaves the tensor untouched apart
            from the inserted axis.

    Returns:
        The same values grouped per foot. Shape is
        (num_envs, num_bodies // bodies_per_foot, bodies_per_foot,...).

    Raises:
        ValueError: If the body count is not a multiple of ``bodies_per_foot``.
    """
    num_bodies = values.shape[1]
    if bodies_per_foot < 1 or num_bodies % bodies_per_foot != 0:
        raise ValueError(
            f"Cannot group {num_bodies} contact bodies into feet of {bodies_per_foot} bodies each:"
            " the selection must be a whole number of feet, in per-foot blocks."
        )
    return values.unflatten(1, (num_bodies // bodies_per_foot, bodies_per_foot))


def foot_group_contact(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, bodies_per_foot: int = 1) -> torch.Tensor:
    """Whether each selected foot is in contact with anything.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the bodies to report, in the requested order.
        bodies_per_foot: Contact bodies making up one foot; a foot is in contact when **any** of them
            is. Defaults to 1, one collider per foot. See :func:`fold_bodies_into_feet`.

    Returns:
        One per foot, 1.0 while in contact. Shape is (num_envs, num_selected_bodies // bodies_per_foot).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    in_contact = (forces.norm(dim=-1) > 0.0).float()
    return fold_bodies_into_feet(in_contact, bodies_per_foot).amax(dim=2)


def foot_group_contact_forces_safe(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, bodies_per_foot: int = 1
) -> torch.Tensor:
    """Net contact force on each selected body, log-compressed and guarded against NaNs.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the bodies to report, in the requested order.
        bodies_per_foot: Contact bodies making up one foot, whose net forces are **summed** into the
            force on the foot. Defaults to 1, one collider per foot. See
            :func:`fold_bodies_into_feet`.

    Returns:
        Compressed contact forces [log1p(N)], flattened per foot.
        Shape is (num_envs, 3 * num_selected_bodies // bodies_per_foot).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    forces = fold_bodies_into_feet(forces, bodies_per_foot).sum(dim=2).flatten(start_dim=1)
    return zero_non_finite(torch.sign(forces) * torch.log1p(forces.abs()))


def foot_group_air_time_safe(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, bodies_per_foot: int = 1
) -> torch.Tensor:
    """Time each selected foot has spent in the air since it last left the ground, guarded.

    See :func:`foot_group_contact_forces_safe` for why the critic's sensor reads are guarded.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the bodies to report, in the requested order. The sensor
            must be configured with ``track_air_time``.
        bodies_per_foot: Contact bodies making up one foot. The foot's air time is the **smallest**
            of theirs, because the foot stops being airborne as soon as any of them lands. Defaults
            to 1, one collider per foot. See :func:`fold_bodies_into_feet`.

    Returns:
        Current air time [s] per foot. Shape is (num_envs, num_selected_bodies // bodies_per_foot).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time
    if air_time is None:
        raise RuntimeError(f"The contact sensor '{sensor_cfg.name}' does not track air time.")
    return zero_non_finite(fold_bodies_into_feet(air_time.torch[:, sensor_cfg.body_ids], bodies_per_foot).amin(dim=2))


def foot_origin_height_safe(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of each selected body above the ground its environment sits on, guarded.

    See :func:`foot_group_contact_forces_safe` for why the critic's sensor reads are guarded.

    Args:
        env: The environment.
        asset_cfg: The articulation and the bodies to report, in the requested order.

    Returns:
        Body heights [m] above the environment origin. Shape is (num_envs, num_selected_bodies).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return zero_non_finite(heights - env.scene.env_origins[:, 2].unsqueeze(-1))


"""
Ball state, for the ball-kick critic.
"""


def ball_pos_in_base(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the ball relative to the robot base, in the base frame.

    Args:
        env: The environment instance.
        asset_cfg: The rigid object to locate.
        robot_cfg: The articulation whose root link frame the position is expressed in.

    Returns:
        The ball position [m] in the base frame. Shape is (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[asset_cfg.name]
    relative_pos_w = ball.data.root_link_pos_w.torch - robot.data.root_link_pos_w.torch
    return zero_non_finite(quat_apply_inverse(robot.data.root_link_quat_w.torch, relative_pos_w))


def ball_vel_in_base(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear velocity of the ball, in the robot base frame.

    Args:
        env: The environment instance.
        asset_cfg: The rigid object whose linear velocity is read.
        robot_cfg: The articulation whose root link frame the velocity is expressed in.

    Returns:
        The ball velocity [m/s] in the base frame. Shape is (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[asset_cfg.name]
    return zero_non_finite(quat_apply_inverse(robot.data.root_link_quat_w.torch, ball.data.root_link_lin_vel_w.torch))
