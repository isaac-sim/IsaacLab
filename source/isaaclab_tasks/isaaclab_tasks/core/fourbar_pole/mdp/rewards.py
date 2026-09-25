# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the fourbar-pole swing-up environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class pole_upright(ManagerTermBase):
    """Pole-uprightness reward that also logs a sustained-upright success metric.

    The reward is ``sum(cos(pole_angle))``: ``+1`` upright and ``-1`` hanging. On reset it flushes
    ``Metrics/success_rate`` into ``extras["log"]``: the fraction of environments that held the pole
    within the upright cone (``cos > success_threshold``) for at least the final ``hold_time_s``
    seconds. Both parameters shape only the metric, not the reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive_upright = torch.zeros(env.num_envs, device=env.device)
        self._success = torch.zeros(env.num_envs, device=env.device)
        hold_time_s: float = cfg.params.get("hold_time_s", 0.5)
        self._hold_steps = max(1, round(hold_time_s / env.step_dt))

    def reset(self, env_ids: torch.Tensor):
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._success[env_ids].mean().item()
        self._consecutive_upright[env_ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, success_threshold: float, hold_time_s: float = 0.5
    ) -> torch.Tensor:
        cos_pole = torch.cos(env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids])
        upright = (cos_pole > success_threshold).all(dim=1)
        self._consecutive_upright = torch.where(
            upright, self._consecutive_upright + 1.0, torch.zeros_like(self._consecutive_upright)
        )
        self._success = (self._consecutive_upright >= self._hold_steps).float()
        return torch.sum(cos_pole, dim=1)
