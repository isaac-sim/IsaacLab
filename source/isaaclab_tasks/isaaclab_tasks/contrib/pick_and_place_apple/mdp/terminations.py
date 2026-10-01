# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode progress and success for the H2 apple handover task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from isaaclab_tasks.contrib.h2_sharpa import advance_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class task_success(ManagerTermBase):
    """Track lift, handover, placement, and release independently of reward weights."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_phase = self.phase.clone()
        self.hold_counter = torch.zeros_like(self.phase)
        self.initial_height = torch.empty(self.num_envs, device=self.device)
        self.left_dist_at_grasp = torch.empty(self.num_envs, device=self.device)
        robot = env.scene[cfg.params["robot_cfg"].name]
        self.wrist_ids, _ = robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True)
        self.thumb_ids, _ = robot.find_joints(
            ["right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_AA", "right_thumb_IP"], preserve_order=True
        )
        self.thumb_thresholds = torch.tensor([2.55, 10.35, 23.4, 18.75], device=self.device) * (torch.pi / 180)

    def reset(self, env_ids=None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.phase[env_ids] = self.previous_phase[env_ids] = self.hold_counter[env_ids] = 0
        self.left_dist_at_grasp[env_ids] = float("inf")
        self.initial_height[env_ids] = self._env.scene[self.cfg.params["apple_cfg"].name].data.root_pos_w[env_ids, 2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        apple_cfg: SceneEntityCfg,
        plate_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        lift_z: float = 0.10,
        grasp_dist: float = 0.26,
        release_dist: float = 0.08,
        place_grasp_dist: float = 0.30,
        xy_radius: float = 0.10,
        z_above: float = 0.025,
        z_window: float = 0.075,
        hold_steps: int = 1,
    ) -> torch.Tensor:
        """Advance phases from current poses; distance thresholds are in [m]."""
        self.previous_phase.copy_(self.phase)
        apple_pos = env.scene[apple_cfg.name].data.root_pos_w
        plate_pos = env.scene[plate_cfg.name].data.root_pos_w
        robot = env.scene[robot_cfg.name]
        left_wrist, right_wrist = robot.data.body_pos_w[:, self.wrist_ids].unbind(dim=1)
        left_dist = torch.linalg.vector_norm(apple_pos - left_wrist, dim=-1)
        right_dist = torch.linalg.vector_norm(apple_pos - right_wrist, dim=-1)
        lifted = apple_pos[:, 2] > self.initial_height + lift_z

        self.phase, self.hold_counter = advance_phase(
            self.phase, self.hold_counter, 0, lifted & (left_dist < grasp_dist), hold_steps
        )
        self.left_dist_at_grasp = torch.where(
            (self.previous_phase == 0) & (self.phase >= 1), left_dist, self.left_dist_at_grasp
        )
        handed_over = lifted & (right_dist < grasp_dist) & (left_dist > self.left_dist_at_grasp + release_dist)
        self.phase, self.hold_counter = advance_phase(self.phase, self.hold_counter, 1, handed_over, hold_steps)

        height = apple_pos[:, 2] - plate_pos[:, 2]
        on_plate = (
            (torch.linalg.vector_norm(apple_pos[:, :2] - plate_pos[:, :2], dim=-1) < xy_radius)
            & (height > z_above)
            & (height < z_above + z_window)
        )
        self.phase, self.hold_counter = advance_phase(
            self.phase, self.hold_counter, 2, on_plate & (right_dist < place_grasp_dist), hold_steps
        )
        thumb_open = (robot.data.joint_pos[:, self.thumb_ids] < self.thumb_thresholds).sum(dim=1) >= 3
        # Placement and release must occur on different steps.
        released = (self.previous_phase == 3) & on_plate & thumb_open
        self.phase, self.hold_counter = advance_phase(self.phase, self.hold_counter, 3, released, hold_steps)
        return self.phase >= 4
