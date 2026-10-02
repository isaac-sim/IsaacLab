# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking observations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply, quat_from_angle_axis

from .events import encoder_bias

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor
_IMU_MISALIGNMENT_ATTR = "_microduck_imu_misalignment"


def joint_pos_rel_biased(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), biased: bool = True
) -> torch.Tensor:
    """Encoder joint offsets from the default pose [rad], in the selected joint order."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if biased:
        joint_pos = joint_pos + encoder_bias(env, asset_cfg)[:, asset_cfg.joint_ids]
    return joint_pos - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


def _imu_misalignment_quat(env: ManagerBasedEnv, max_angle_rad: float) -> torch.Tensor:
    """The per-environment IMU mounting-misalignment rotation, sampled once per run."""
    cached: tuple[float, torch.Tensor] | None = getattr(env, _IMU_MISALIGNMENT_ATTR, None)
    if cached is not None:
        angle, quat = cached
        if angle != max_angle_rad:
            raise ValueError("IMU observations must use the same maximum misalignment angle.")
        return quat
    axis = torch.randn(env.num_envs, 3, device=env.device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-08)
    angles = torch.rand(env.num_envs, device=env.device) * max_angle_rad
    quat = quat_from_angle_axis(angles, axis)
    setattr(env, _IMU_MISALIGNMENT_ATTR, (max_angle_rad, quat))
    return quat


def projected_gravity_imu_misaligned(
    env: ManagerBasedEnv, max_angle_deg: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity projected on the root frame, seen through a misaligned IMU."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(quat, asset.data.projected_gravity_b.torch)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedEnv, max_angle_deg: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity [rad/s], rotated by the same IMU misalignment as gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(quat, asset.data.root_ang_vel_b.torch)


class delayed_observation(ManagerTermBase):
    """Wraps another observation term in the stochastic bus latency upstream models."""

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
        self._pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_pending_reset = False
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
        """Compute the wrapped term and return a stale copy of it."""
        step = getattr(env, "common_step_counter", None)
        if step is not None and step == self._last_step and (self._output is not None):
            if not self._has_pending_reset:
                return self._output
            fresh = term_func(env, **term_params or {})
            mask = self._pending_reset.view(-1, *[1] * (fresh.dim() - 1))
            return torch.where(mask, fresh, self._output)
        obs = term_func(env, **term_params or {})
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


def _finite(value: torch.Tensor) -> torch.Tensor:
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
    return _finite(torch.sign(forces) * torch.log1p(forces.abs()))


def foot_air_time_safe(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Air time [s] per foot, with non-finite values replaced by zero."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time
    if air_time is None:
        raise RuntimeError(f"The contact sensor '{sensor_cfg.name}' does not track air time.")
    return _finite(air_time.torch[:, sensor_cfg.body_ids])


def foot_height_safe(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot height [m] above the flat ground, with non-finite values replaced by zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return _finite(heights - env.scene.env_origins[:, 2].unsqueeze(-1))
