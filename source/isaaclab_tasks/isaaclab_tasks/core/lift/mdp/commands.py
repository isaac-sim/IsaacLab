# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum-aware commands for lift tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.utils.configclass import configclass

from .curriculums import lift_difficulty_fraction


class CurriculumPoseCommand(UniformPoseCommand):
    """Blend a fixed easy goal into the full uniformly sampled goal distribution."""

    cfg: CurriculumPoseCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        difficulty = lift_difficulty_fraction(self._env, env_ids)
        blend = (difficulty / self.cfg.full_goal_difficulty).clamp(0.0, 1.0).unsqueeze(-1)
        easy_goal = torch.tensor(self.cfg.easy_goal, device=self.device)
        self.pose_command_b[env_ids, :3] = torch.lerp(easy_goal, self.pose_command_b[env_ids, :3], blend)


@configclass
class CurriculumPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`CurriculumPoseCommand`."""

    class_type: type[CurriculumPoseCommand] = CurriculumPoseCommand

    easy_goal: tuple[float, float, float] = (0.5, 0.0, 0.35)
    """Goal position used at zero difficulty in the robot root frame [m]."""

    full_goal_difficulty: float = 0.3
    """Difficulty fraction at which the full goal distribution is active."""
