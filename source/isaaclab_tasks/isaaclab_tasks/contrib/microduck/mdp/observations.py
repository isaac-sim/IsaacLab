# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking observations."""

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
    """Root angular velocity [rad/s] then projected gravity, read by a misaligned IMU with bus latency.

    Each environment draws a fixed mounting rotation at construction, about a uniform random axis by an angle in
    ``[0, max_angle_deg]`` [deg], and applies it to both readings. Each reading lags by its own draw in
    ``[min_lag, max_lag]`` samples, redrawn every ``update_period`` samples with its own per-environment phase to
    match upstream's separate IMU delay buffers. All parameters are fixed at construction; ``max_lag`` sizes the
    history buffer.
    """

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
