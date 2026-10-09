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
from isaaclab.utils import env_mask_from_ids

from isaaclab_tasks.core.reorient.utils import EpisodeErrorRecorder

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import HandoverCommandCfg


class HandoverCommand(CommandTerm):
    """Sample the fixed-position, random-orientation handover goal pose."""

    cfg: HandoverCommandCfg

    def __init__(self, cfg: HandoverCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.asset_name]
        offset = torch.tensor(cfg.position_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self._object.data.default_root_pose.torch[:, :3] + offset
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self._minimum_goal_distance = EpisodeErrorRecorder(self.num_envs, self.device)
        # Whether each environment has brought the object within the success distance at any point
        # this episode. Necessary but not sufficient for success; see ``reset``.
        self._succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Goal pose in the environment frame [m, unit quaternion]. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    def _update_metrics(self) -> None:
        object_pos = self._object.data.root_pos_w.torch - self._env.scene.env_origins
        goal_distance = torch.linalg.norm(object_pos - self.pos_command_e, ord=2, dim=-1)
        self.metrics["goal_distance"][:] = goal_distance
        self._minimum_goal_distance.update(goal_distance)
        self._succeeded |= goal_distance < self.cfg.success_distance_threshold

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: torch.Tensor | None = None) -> dict[str, float]:
        if env_mask is None:
            env_mask = env_mask_from_ids(env_ids, self.num_envs, self.device)
        # The base class averages the metric over the reset envs and zeroes it, so the episode's success bit is
        # written before delegating. Success means the object is at the goal when the episode ends, not that
        # it passed through it; the latch guards the first reset, before any distance is measured.
        success = (self.metrics["goal_distance"] < self.cfg.success_distance_threshold) & self._succeeded
        success_rate = self.metrics["success_rate"]
        torch.where(env_mask, success.float(), success_rate, out=success_rate)
        extras = super().reset(env_mask=env_mask)
        self._succeeded.masked_fill_(env_mask, False)
        log = self._env.extras.setdefault("log", {})
        # Route success_rate to the unified ``Metrics/success_rate`` path (shared TensorBoard
        # card across tasks); pop it from the returned dict so CommandManager does not
        # additionally log it under ``Metrics/<term_name>/success_rate``.
        log["Metrics/success_rate"] = extras.pop("success_rate")
        for statistic, value in self._minimum_goal_distance.reset(env_mask).items():
            log[f"Diagnostics/episode_min_goal_distance_{statistic}"] = value
        return extras

    def _resample_command(self, env_mask: torch.Tensor) -> None:
        # sample uniformly over SO(3) rather than composing single-axis rotations, which only reaches a subset
        quat = math_utils.random_orientation(self.num_envs, device=self.device)
        torch.where(env_mask[:, None], quat, self.quat_command_w, out=self.quat_command_w)

    def _update_command(self) -> None:
        pass

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
