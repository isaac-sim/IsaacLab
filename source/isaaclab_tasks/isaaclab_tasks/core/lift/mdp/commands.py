# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum-aware commands for lift tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

from .curriculums import lift_difficulty_fraction

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


class CurriculumPoseCommand(UniformPoseCommand):
    """Blend a fixed easy goal into the full uniformly sampled goal distribution."""

    cfg: CurriculumPoseCommandCfg

    def __init__(self, cfg: CurriculumPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.object: RigidObject = env.scene[cfg.tracked_object_name]

    def _update_metrics(self) -> None:
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_pos_w.torch,
            self.object.data.root_quat_w.torch,
        )
        self.metrics["position_error"] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.linalg.norm(rot_error, dim=-1)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self.robot.is_initialized or not self.object.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        self.current_pose_visualizer.visualize(
            self.object.data.root_pos_w.torch,
            self.object.data.root_quat_w.torch,
        )

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

    tracked_object_name: str = "object"
    """Scene entity whose pose is used for command metrics and visualization."""

    easy_goal: tuple[float, float, float] = (0.5, 0.0, 0.35)
    """Goal position used at zero difficulty in the robot root frame [m]."""

    full_goal_difficulty: float = 0.3
    """Difficulty fraction at which the full goal distribution is active."""
