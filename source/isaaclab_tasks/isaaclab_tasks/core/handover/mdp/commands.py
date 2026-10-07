# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-pose command for the manager-based handover task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from isaaclab_tasks.core.reorient.utils import EpisodeErrorRecorder

from ..handover_common import HandoverGoal

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import HandoverCommandCfg


class HandoverCommand(CommandTerm):
    """Alternate handover goals after cumulative dwell at each receiving hand."""

    cfg: HandoverCommandCfg

    def __init__(self, cfg: HandoverCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.asset_name]
        self._goal = HandoverGoal(
            torch.stack(
                (
                    env.scene[cfg.right_hand_name].data.default_root_pose.torch,
                    env.scene[cfg.left_hand_name].data.default_root_pose.torch,
                )
            ),
            cfg.position_offset,
        )
        self.pos_command_e = self._goal.position
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self._minimum_goal_distance = EpisodeErrorRecorder(self.num_envs, self.device)

    @property
    def command(self) -> torch.Tensor:
        """Goal pose in the environment frame [m, unit quaternion]. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    def _update_metrics(self) -> None:
        object_pos = self._object.data.root_pos_w.torch - self._env.scene.env_origins
        goal_distance = torch.linalg.norm(object_pos - self.pos_command_e, ord=2, dim=-1)
        self.metrics["goal_distance"][:] = goal_distance
        self._minimum_goal_distance.update(goal_distance)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        goals = self._goal.success.snapshot(env_ids)
        self.metrics["success_rate"][env_ids] = goals / (goals + 1.0)
        self.metrics["consecutive_success"][env_ids] = goals
        extras = super().reset(env_ids)
        log = self._env.extras.setdefault("log", {})
        log["Metrics/success_rate"] = extras.pop("success_rate")
        log["Metrics/consecutive_success"] = extras.pop("consecutive_success")
        for statistic, value in self._minimum_goal_distance.reset(env_ids).items():
            log[f"Diagnostics/episode_min_goal_distance_{statistic}"] = value
        return extras

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        self._goal.reset(env_ids)
        # sample uniformly over SO(3) rather than composing single-axis rotations, which only reaches a subset
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        self.quat_command_w[env_ids] = math_utils.random_orientation(num_envs, device=self.device)

    def _update_command(self) -> None:
        succeeded = self.metrics["goal_distance"] < self.cfg.success_distance_threshold
        # Commands follow autoreset; the reset pose must not earn dwell this step.
        succeeded &= ~(self._env.reset_terminated | self._env.reset_time_outs)
        self._goal.update(succeeded, self.cfg.success_dwell_steps)
        self._env.extras.setdefault("log", {})["Diagnostics/dwell_steps"] = self._goal.dwell.float().mean()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):
                self._goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self._goal_visualizer.set_visibility(True)
        elif hasattr(self, "_goal_visualizer"):
            self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        self._goal_visualizer.visualize(
            translations=self.pos_command_e + self._env.scene.env_origins,
            orientations=self.quat_command_w,
        )
