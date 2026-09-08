# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg import CabinetDirectEnvCfg


class CabinetDirectEnv(DirectRLEnv):
    """Direct-workflow environment for opening a cabinet drawer with a parallel-jaw gripper."""

    cfg: CabinetDirectEnvCfg

    def __init__(self, cfg: CabinetDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_joint_ids, _ = self._robot.find_joints(self.cfg.arm_joint_names)
        self.finger_joint_ids, _ = self._robot.find_joints(self.cfg.finger_joint_names)
        self.ee_body_idx = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self.left_finger_body_idx = self._robot.find_bodies(self.cfg.left_finger_body_name)[0][0]
        self.right_finger_body_idx = self._robot.find_bodies(self.cfg.right_finger_body_name)[0][0]
        self.drawer_handle_body_idx = self._cabinet.find_bodies(self.cfg.drawer_handle_body_name)[0][0]
        self.drawer_joint_idx = self._cabinet.find_joints(self.cfg.drawer_joint_name)[0][0]

        expected_action_space = len(self.arm_joint_ids) + 1
        if self.cfg.action_space != expected_action_space:
            raise ValueError(
                f"{type(self.cfg).__name__} declares action_space={self.cfg.action_space}, but its configured arm"
                f" joints and binary gripper require action_space={expected_action_space}."
            )
        expected_observation_space = 2 * self._robot.num_joints + 2 + 3 + expected_action_space
        if self.cfg.observation_space != expected_observation_space:
            raise ValueError(
                f"{type(self.cfg).__name__} declares observation_space={self.cfg.observation_space}, but the"
                f" manager-equivalent observation requires observation_space={expected_observation_space}."
            )

        self.previous_actions = torch.zeros_like(self.actions)
        self.arm_joint_targets = torch.zeros((self.num_envs, len(self.arm_joint_ids)), device=self.device)
        self.finger_joint_targets = torch.zeros((self.num_envs, len(self.finger_joint_ids)), device=self.device)

        def _repeat(value: tuple[float, ...]) -> torch.Tensor:
            return torch.tensor(value, device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        self.ee_pos_offset = _repeat(self.cfg.ee_pos_offset)
        self.finger_pos_offset = _repeat(self.cfg.finger_pos_offset)
        self.drawer_handle_pos_offset = _repeat(self.cfg.drawer_handle_pos_offset)
        self.drawer_handle_rot_offset = _repeat(self.cfg.drawer_handle_rot_offset)

        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.left_finger_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_finger_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_handle_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_handle_quat_w = torch.zeros((self.num_envs, 4), device=self.device)

        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._best_drawer_pos = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "approach_ee_handle",
                "align_ee_handle",
                "approach_gripper_handle",
                "align_grasp_around_handle",
                "grasp_handle",
                "open_drawer_bonus",
                "multi_stage_open_drawer",
                "action_rate_l2",
                "joint_vel",
            )
        }

    def _setup_scene(self) -> None:
        self._robot = self.scene["robot"]
        self._cabinet = self.scene["cabinet"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions[:] = self.actions
        self.actions[:] = actions

        self.arm_joint_targets[:] = (
            self._robot.data.default_joint_pos.torch[:, self.arm_joint_ids]
            + self.cfg.arm_action_scale * self.actions[:, : len(self.arm_joint_ids)]
        )
        self.finger_joint_targets[:] = torch.where(
            self.actions[:, -1:] < 0.0,
            self.cfg.gripper_close_command,
            self.cfg.gripper_open_command,
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self.arm_joint_targets,
            joint_ids=self.arm_joint_ids,
        )
        self._robot.set_joint_position_target_index(
            target=self.finger_joint_targets,
            joint_ids=self.finger_joint_ids,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        drawer_pos = self._cabinet.data.joint_pos.torch[:, self.drawer_joint_idx]
        self._episode_succeeded |= drawer_pos > self.cfg.success_drawer_pos_threshold
        self._best_drawer_pos = torch.maximum(self._best_drawer_pos, drawer_pos)

        distance = torch.linalg.norm(self.drawer_handle_pos_w - self.ee_pos_w, dim=-1, ord=2)
        approach_ee_handle = 1.0 / (1.0 + distance**2)
        approach_ee_handle = approach_ee_handle**2
        approach_ee_handle = torch.where(
            distance <= self.cfg.approach_ee_handle_threshold,
            2.0 * approach_ee_handle,
            approach_ee_handle,
        )

        ee_rot_mat = matrix_from_quat(self.ee_quat_w)
        handle_rot_mat = matrix_from_quat(self.drawer_handle_quat_w)
        handle_x, handle_y = handle_rot_mat[..., 0], handle_rot_mat[..., 1]
        ee_x, ee_z = ee_rot_mat[..., 0], ee_rot_mat[..., 2]
        align_z = torch.sum(ee_z * -handle_x, dim=-1)
        align_x = torch.sum(ee_x * -handle_y, dim=-1)
        align_ee_handle = 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)

        left_finger_distance = torch.abs(self.left_finger_pos_w[:, 2] - self.drawer_handle_pos_w[:, 2])
        right_finger_distance = torch.abs(self.right_finger_pos_w[:, 2] - self.drawer_handle_pos_w[:, 2])
        graspable = (self.right_finger_pos_w[:, 2] < self.drawer_handle_pos_w[:, 2]) & (
            self.left_finger_pos_w[:, 2] > self.drawer_handle_pos_w[:, 2]
        )
        align_grasp_around_handle = graspable.float()
        approach_gripper_handle = align_grasp_around_handle * (
            2.0 * self.cfg.approach_gripper_handle_offset - left_finger_distance - right_finger_distance
        )

        finger_joint_pos = self._robot.data.joint_pos.torch[:, self.finger_joint_ids]
        grasp_handle = (distance <= self.cfg.grasp_handle_threshold) * torch.sum(
            self.cfg.gripper_open_command - finger_joint_pos,
            dim=-1,
        )
        open_drawer = (align_grasp_around_handle + 1.0) * drawer_pos
        multi_stage_open_drawer = (
            (drawer_pos > 0.01).float() * 0.5
            + (drawer_pos > 0.2).float() * align_grasp_around_handle
            + (drawer_pos > 0.3).float() * align_grasp_around_handle
        )

        action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel.torch), dim=-1)

        reward_terms = {
            "approach_ee_handle": self.cfg.approach_ee_handle_reward_scale * approach_ee_handle,
            "align_ee_handle": self.cfg.align_ee_handle_reward_scale * align_ee_handle,
            "approach_gripper_handle": self.cfg.approach_gripper_handle_reward_scale * approach_gripper_handle,
            "align_grasp_around_handle": (self.cfg.align_grasp_around_handle_reward_scale * align_grasp_around_handle),
            "grasp_handle": self.cfg.grasp_handle_reward_scale * grasp_handle,
            "open_drawer_bonus": self.cfg.open_drawer_reward_scale * open_drawer,
            "multi_stage_open_drawer": (self.cfg.multi_stage_open_drawer_reward_scale * multi_stage_open_drawer),
            "action_rate_l2": self.cfg.action_rate_reward_scale * action_rate,
            "joint_vel": self.cfg.joint_vel_reward_scale * joint_vel,
        }
        reward = torch.zeros_like(drawer_pos)
        for name, value in reward_terms.items():
            value = value * self.step_dt
            reward += value
            self._episode_reward_sums[name] += value
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
        log["Metrics/drawer_pos"] = self._best_drawer_pos[env_ids].mean().item()
        for name, episode_sum in self._episode_reward_sums.items():
            log[f"Episode_Reward/{name}"] = torch.mean(episode_sum[env_ids]) / self.max_episode_length_s
            episode_sum[env_ids] = 0.0

        super()._reset_idx(env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self._episode_succeeded[env_ids] = False
        self._best_drawer_pos[env_ids] = 0.0
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        robot_joint_pos = self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch
        robot_joint_vel = self._robot.data.joint_vel.torch - self._robot.data.default_joint_vel.torch
        drawer_joint_pos = (
            self._cabinet.data.joint_pos.torch[:, self.drawer_joint_idx]
            - self._cabinet.data.default_joint_pos.torch[:, self.drawer_joint_idx]
        ).unsqueeze(-1)
        drawer_joint_vel = (
            self._cabinet.data.joint_vel.torch[:, self.drawer_joint_idx]
            - self._cabinet.data.default_joint_vel.torch[:, self.drawer_joint_idx]
        ).unsqueeze(-1)
        relative_ee_drawer_distance = self.drawer_handle_pos_w - self.ee_pos_w

        observation = torch.cat(
            (
                robot_joint_pos,
                robot_joint_vel,
                drawer_joint_pos,
                drawer_joint_vel,
                relative_ee_drawer_distance,
                torch.clamp(self.actions, -5.0, 5.0),
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        ee_body_pos = self._robot.data.body_pos_w.torch[env_ids, self.ee_body_idx]
        ee_body_quat = self._robot.data.body_quat_w.torch[env_ids, self.ee_body_idx]
        self.ee_pos_w[env_ids], self.ee_quat_w[env_ids] = combine_frame_transforms(
            ee_body_pos,
            ee_body_quat,
            self.ee_pos_offset[env_ids],
        )

        left_finger_body_pos = self._robot.data.body_pos_w.torch[env_ids, self.left_finger_body_idx]
        left_finger_body_quat = self._robot.data.body_quat_w.torch[env_ids, self.left_finger_body_idx]
        self.left_finger_pos_w[env_ids], _ = combine_frame_transforms(
            left_finger_body_pos,
            left_finger_body_quat,
            self.finger_pos_offset[env_ids],
        )

        right_finger_body_pos = self._robot.data.body_pos_w.torch[env_ids, self.right_finger_body_idx]
        right_finger_body_quat = self._robot.data.body_quat_w.torch[env_ids, self.right_finger_body_idx]
        self.right_finger_pos_w[env_ids], _ = combine_frame_transforms(
            right_finger_body_pos,
            right_finger_body_quat,
            self.finger_pos_offset[env_ids],
        )

        drawer_handle_body_pos = self._cabinet.data.body_pos_w.torch[env_ids, self.drawer_handle_body_idx]
        drawer_handle_body_quat = self._cabinet.data.body_quat_w.torch[env_ids, self.drawer_handle_body_idx]
        self.drawer_handle_pos_w[env_ids], self.drawer_handle_quat_w[env_ids] = combine_frame_transforms(
            drawer_handle_body_pos,
            drawer_handle_body_quat,
            self.drawer_handle_pos_offset[env_ids],
            self.drawer_handle_rot_offset[env_ids],
        )
