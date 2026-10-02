# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor


def joint_pos_rel_biased(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    biased: bool = True,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Encoder joint offsets from the default pose [rad].

    ``asset_cfg`` must select the joints of the ``action_name`` action term, in the same order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if biased:
        joint_pos = joint_pos + env.action_manager.get_term(action_name).encoder_bias
    return joint_pos - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


def projected_gravity_imu_misaligned(
    env: ManagerBasedEnv, event_name: str = "imu_misalignment", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity projected on the root frame, seen through the misaligned IMU of event ``event_name``."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = env.event_manager.get_term_cfg(event_name).func.quat
    return quat_apply(quat, asset.data.projected_gravity_b.torch)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedEnv, event_name: str = "imu_misalignment", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity [rad/s], seen through the misaligned IMU of event ``event_name``."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = env.event_manager.get_term_cfg(event_name).func.quat
    return quat_apply(quat, asset.data.root_ang_vel_b.torch)


class delayed_observation(ManagerTermBase):
    """Wraps another observation term in a lag redrawn on a fixed period with a per-environment phase.

    This matches upstream's IMU bus latency; constant lags use :attr:`ObservationTermCfg.delay_min_lag`.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._min_lag = cfg.params["min_lag"]
        self._max_lag = cfg.params["max_lag"]
        self._update_period = cfg.params["update_period"]
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
        phases = torch.randint(0, self._update_period, (self.num_envs,), device=self.device)
        self._phase_offsets[resolved] = phases[resolved]
        self._pending_reset[resolved] = True
        self._has_pending_reset = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        term_func: Callable[[ManagerBasedRLEnv], torch.Tensor],
        min_lag: int,
        max_lag: int,
        update_period: int,
    ) -> torch.Tensor:
        """Compute the wrapped term and return a stale copy of it."""
        # Repeated reads within one step return the same sample, except for freshly reset environments.
        if env.common_step_counter == self._last_step and self._output is not None:
            if not self._has_pending_reset:
                return self._output
            fresh = term_func(env)
            mask = self._pending_reset.view(-1, *[1] * (fresh.dim() - 1))
            return torch.where(mask, fresh, self._output)
        obs = term_func(env)
        if self._has_pending_reset:
            self._buffer.reset(self._pending_reset)
            self._pending_reset.fill_(False)
            self._has_pending_reset = False
        self._last_step = env.common_step_counter
        self._buffer.append(obs)
        should_update = (self._step_count + self._phase_offsets) % self._update_period == 0
        candidates = torch.randint(min_lag, max_lag + 1, (self.num_envs,), device=self.device)
        self._lags = torch.where(should_update, candidates, self._lags)
        self._step_count += 1
        self._output = self._buffer[self._lags]
        return self._output


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
    return _finite(sensor.data.current_air_time.torch[:, sensor_cfg.body_ids])


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
    return _finite(foot_height(env, asset_cfg, height_sensor_names))
