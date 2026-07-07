# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum-aware actions for lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.configclass import configclass

from .curriculums import lift_difficulty_fraction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CurriculumGripperAction(BinaryJointPositionAction):
    """Keep a pre-grasped curriculum reset closed until grasp learning begins."""

    cfg: CurriculumGripperActionCfg

    def __init__(self, cfg: CurriculumGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env_ids = torch.arange(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply policy actions, overriding early pre-grasped stages with close."""
        super().process_actions(actions)
        difficulty = lift_difficulty_fraction(self._env, self._env_ids)
        force_close = difficulty < self.cfg.force_close_below_difficulty
        self._processed_actions[force_close] = self._close_command


class CurriculumDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Ramp task-space control authority with the goal-distribution difficulty."""

    cfg: CurriculumDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: CurriculumDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env_ids = torch.arange(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale physical arm motion while preserving raw policy actions for learning."""
        difficulty = lift_difficulty_fraction(self._env, self._env_ids)
        control_scale = (difficulty / self.cfg.full_control_difficulty).clamp(0.0, 1.0).unsqueeze(-1)
        processed_actions = actions * control_scale
        super().process_actions(processed_actions)
        self._raw_actions[:] = actions


@configclass
class CurriculumGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration for :class:`CurriculumGripperAction`."""

    class_type: type[CurriculumGripperAction] = CurriculumGripperAction

    force_close_below_difficulty: float = 0.45
    """Normalized difficulty below which the physical gripper stays closed."""


@configclass
class CurriculumDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`CurriculumDifferentialInverseKinematicsAction`."""

    class_type: type[CurriculumDifferentialInverseKinematicsAction] = CurriculumDifferentialInverseKinematicsAction

    full_control_difficulty: float = 0.3
    """Normalized difficulty at which policy arm commands reach their configured scale."""
