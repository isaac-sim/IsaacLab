# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


class survival_success_rate(ManagerTermBase):
    """Tracks episode survival as the success metric.

    Returns zero reward (pure metric tracking). Flushes ``Metrics/success_rate``
    into ``extras["log"]`` on episode reset, where success = timed out without
    early termination.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor):
        survived = self._env.termination_manager.time_outs[env_ids]
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = survived.float().mean().item()

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)
