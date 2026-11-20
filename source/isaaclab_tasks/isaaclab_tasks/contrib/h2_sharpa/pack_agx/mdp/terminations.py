# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode progress and success for packing an AGX Orin into its box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_tasks.contrib.h2_sharpa.mdp import advance_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class task_success(ManagerTermBase):
    """Track lift, alignment, seating, and release independently of reward weights."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_phase = self.phase.clone()
        self.hold_counter = torch.zeros_like(self.phase)
        self.initial_height = torch.empty(self.num_envs, device=self.device)

    def reset(self, env_ids=None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.phase[env_ids] = self.previous_phase[env_ids] = self.hold_counter[env_ids] = 0
        self.initial_height[env_ids] = self._env.scene[self.cfg.params["agx_orin_cfg"].name].data.root_pos_w[env_ids, 2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        agx_orin_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        protective_box_cfg: SceneEntityCfg,
        lift_z: float = 0.01,
        align_target_z_offset: float = 0.10,
        align_xy: float = 0.025,
        rotation_tolerance: float = 0.2,
        lift_hold_steps: int = 5,
        align_hold_steps: int = 5,
        seat_hold_steps: int = 5,
    ) -> torch.Tensor:
        """Advance phases using distances [m] and roll/pitch tolerances [rad]."""
        self.previous_phase.copy_(self.phase)
        agx = env.scene[agx_orin_cfg.name]
        robot = env.scene[robot_cfg.name]
        position = agx.data.root_pos_w
        delta = position - env.scene[protective_box_cfg.name].data.root_pos_w
        roll, pitch, _ = euler_xyz_from_quat(agx.data.root_quat_w.torch)
        horizontal = (roll.abs() <= rotation_tolerance) & (pitch.abs() <= rotation_tolerance)
        lifted = (position[:, 2] > self.initial_height + lift_z) & horizontal
        self.phase, self.hold_counter = advance_phase(self.phase, self.hold_counter, 0, lifted, lift_hold_steps)

        alignment = delta.clone()
        alignment[:, 2] -= align_target_z_offset
        aligned = (self.previous_phase == 1) & lifted & (torch.linalg.vector_norm(alignment, dim=-1) <= align_xy)
        self.phase, self.hold_counter = advance_phase(self.phase, self.hold_counter, 1, aligned, align_hold_steps)

        in_box = (delta[:, :2].abs() <= 0.01).all(dim=-1) & (delta[:, 2].abs() <= 0.03)
        in_box &= (roll.abs() <= 0.2) & (pitch.abs() <= 0.2)
        seated = (self.previous_phase == 2) & in_box
        self.phase, self.hold_counter = advance_phase(self.phase, self.hold_counter, 2, seated, seat_hold_steps)
        wrist_pos = robot.data.body_pos_w.torch[:, robot_cfg.body_ids].squeeze(1)
        wrist_distance = torch.linalg.vector_norm(wrist_pos - position, dim=-1)
        released = (
            (self.previous_phase == 3)
            & in_box
            & (robot.data.joint_pos.torch[:, robot_cfg.joint_ids].squeeze(1) <= 0.261799)
            & (wrist_distance >= 0.32)
        )
        self.phase, self.hold_counter = advance_phase(self.phase, self.hold_counter, 3, released, seat_hold_steps)
        return self.phase >= 4
