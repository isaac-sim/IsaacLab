# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Position commands with bounded target motion for cabinet manipulation."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import RateLimitedJointPositionActionCfg


class RateLimitedJointPositionAction(JointPositionAction):
    """Limit position-target changes per policy step, starting from the reset joint pose.

    This bounds commanded motion, not the physical velocity resulting from contact or tracking error.
    """

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not isfinite(cfg.max_velocity) or cfg.max_velocity <= 0.0:
            raise ValueError("max_velocity must be finite and positive.")
        self._max_delta = cfg.max_velocity * env.step_dt
        self._target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._target.add_((self._processed_actions - self._target).clamp(-self._max_delta, self._max_delta))
        self._processed_actions.copy_(self._target)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._target[env_ids] = self._asset.data.joint_pos.torch[:, self._joint_ids][env_ids]
        self._processed_actions[env_ids] = self._target[env_ids]
