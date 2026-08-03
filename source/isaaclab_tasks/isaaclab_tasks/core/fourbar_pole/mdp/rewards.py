# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the fourbar-pole swing-up environments.

This module holds both observation helpers (``joint_pos_cos`` / ``joint_pos_sin``)
that encode an angle without the ``+-pi`` wrap discontinuity, and the swing-up
reward / success-metric terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cosine of the selected joint positions.

    Encodes the angle without the wrap-around discontinuity at ``+-pi`` so the
    policy sees a smooth signal as the pole swings through the bottom.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cos(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


def joint_pos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sine of the selected joint positions (companion to :func:`joint_pos_cos`)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sin(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


class pole_upright(ManagerTermBase):
    """Pole-uprightness reward that also logs a sustained-upright success metric.

    Reward is ``sum(cos(pole_angle))`` -- ``+1`` upright, ``-1`` hanging. On reset it flushes
    ``Metrics/success_rate``: the fraction of environments that held the pole within the upright
    cone (``cos > success_threshold``) for at least the final ``hold_time_s`` seconds. Both params
    shape only the metric, not the reward.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
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
