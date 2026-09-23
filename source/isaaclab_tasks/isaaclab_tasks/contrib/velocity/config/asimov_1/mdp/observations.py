# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FOOT_SITE_OFFSET = (0.05, 0.0, -0.025)

_DEFAULT_FEET_CFG = SceneEntityCfg(
    "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
)


def foot_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    offset: tuple[float, float, float] = FOOT_SITE_OFFSET,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    off = torch.tensor(offset, device=env.device).expand(pos.shape[0], pos.shape[1], 3)
    return pos + math_utils.quat_apply(quat, off)


def foot_vel_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    offset: tuple[float, float, float] = FOOT_SITE_OFFSET,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    lin_vel = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]
    off = torch.tensor(offset, device=env.device).expand(quat.shape[0], quat.shape[1], 3)
    r = math_utils.quat_apply(quat, off)
    return lin_vel + torch.cross(ang_vel, r, dim=-1)


def foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    return foot_pos_w(env, asset_cfg)[:, :, 2]


def foot_air_time(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    return sensor.data.current_air_time.torch


def foot_contact(env: ManagerBasedRLEnv, sensor_name: str, force_threshold: float = 1.0) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    return (sensor.data.net_forces_w.torch.norm(dim=-1) > force_threshold).float()


def foot_contact_forces(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    return sensor.data.net_forces_w.torch.reshape(env.num_envs, -1)


class delayed_obs(ManagerTermBase):
    """Observation term that applies a randomized per-environment sensor lag."""

    _QUANTITIES = {
        "base_ang_vel": base_mdp.base_ang_vel,
        "projected_gravity": base_mdp.projected_gravity,
    }

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._min_lag = int(cfg.params["min_lag"])
        self._max_lag = int(cfg.params["max_lag"])
        self._fn = self._QUANTITIES[cfg.params["quantity"]]
        self._lags = torch.randint(self._min_lag, self._max_lag + 1, (env.num_envs,), device=env.device)
        self._buffer: torch.Tensor | None = None
        self._pending_reset: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._lags[env_ids] = torch.randint(self._min_lag, self._max_lag + 1, (len(env_ids),), device=self._env.device)
        self._pending_reset = env_ids

    def __call__(self, env: ManagerBasedRLEnv, quantity: str, min_lag: int, max_lag: int) -> torch.Tensor:
        value = self._fn(env)
        if self._buffer is None:
            self._buffer = value.unsqueeze(1).repeat(1, self._max_lag + 1, 1)
        else:
            self._buffer = torch.roll(self._buffer, shifts=1, dims=1)
            self._buffer[:, 0] = value
        if self._pending_reset is not None:
            self._buffer[self._pending_reset] = value[self._pending_reset].unsqueeze(1)
            self._pending_reset = None
        env_ids = torch.arange(env.num_envs, device=env.device)
        return self._buffer[env_ids, self._lags]
