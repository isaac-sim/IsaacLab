# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent Inspire finger commands with bounded joint-target changes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions import BinaryJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

    from .inspire_hand_cfg import InspireFingerActionCfg


class InspireFingerAction(BinaryJointPositionAction):
    """Select open/close finger targets and advance toward them at a bounded angular rate.

    The binary action controls the configured finger chains together. Joint targets advance once per
    control step, independently of whether the physics backend handles decimation internally.
    """

    cfg: InspireFingerActionCfg

    def __init__(self, cfg: InspireFingerActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if not math.isfinite(cfg.max_target_velocity) or cfg.max_target_velocity <= 0.0:
            raise ValueError("max_target_velocity must be finite and positive")
        self._joint_indices = wp.to_torch(self._joint_ids).long()
        self._delta = self._close_command - self._open_command
        self._delta_norm_squared = float(self._delta.square().sum().item())
        if self._delta_norm_squared == 0.0:
            raise ValueError("Finger open and close commands must differ")
        self._maximum_progress_rate = cfg.max_target_velocity / float(self._delta.abs().max().item())
        self._progress = torch.zeros((self.num_envs, 1), device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        desired = ((self._processed_actions - self._open_command) * self._delta).sum(dim=1, keepdim=True)
        desired.div_(self._delta_norm_squared)
        maximum_change = self._maximum_progress_rate * self._env.step_dt
        self._progress.add_((desired - self._progress).clamp(-maximum_change, maximum_change))
        # A shared interpolation fraction preserves authored mimic ratios throughout the movement.
        self._processed_actions.copy_(self._open_command + self._progress * self._delta)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        positions = self._asset.data.joint_pos.torch[indices][:, self._joint_indices]
        progress = ((positions - self._open_command) * self._delta).sum(dim=1, keepdim=True)
        self._progress[indices] = (progress / self._delta_norm_squared).clamp(0.0, 1.0)
        self._processed_actions[indices] = self._open_command + self._progress[indices] * self._delta


def inspire_finger_tracking_error(
    env: ManagerBasedEnv,
    finger_cfgs: tuple[SceneEntityCfg, ...],
    motion_ranges: tuple[float, ...],
) -> torch.Tensor:
    """Return signed target errors for each thumb/index primary joint, normalized by its stroke."""
    errors = []
    for cfg, stroke in zip(finger_cfgs, motion_ranges, strict=True):
        robot = env.scene[cfg.name]
        joint = cfg.joint_ids[0]
        errors.append((robot.data.joint_pos_target.torch[:, joint] - robot.data.joint_pos.torch[:, joint]) / stroke)
    return torch.stack(errors, dim=1)
