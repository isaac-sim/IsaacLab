# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import carb
import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, sample_uniform, saturate

from .allegro_rotate_env_cfg import allegro_grasp_cache_path

if TYPE_CHECKING:
    from .allegro_rotate_env_cfg import AllegroRotateEnvCfg


class AllegroRotateEnv(DirectRLEnv):
    """Allegro palm-supported free-cylinder rolling task.

    This is a free-object rotation task: no screw joint and no thread collision.
    The object is allowed to be supported by the hand cradle. The reward focuses
    on angular progress, object motion penalty, joint-pose penalty, torque/work
    penalties, and object position retention.
    """

    cfg: AllegroRotateEnvCfg

    def __init__(self, cfg: AllegroRotateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        self.num_hand_dofs = self.hand.num_joints

        self.actuated_dof_indices = []
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        self.finger_bodies = []
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.fingertip_log_names = [body_name.replace("_link_3", "") for body_name in self.cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)
        self.thumb_finger_id = self.fingertip_log_names.index("thumb")
        self.ring_finger_id = self.fingertip_log_names.index("ring")
        self.non_thumb_finger_ids = torch.tensor(
            [finger_id for finger_id in range(self.num_fingertips) if finger_id != self.thumb_finger_id],
            dtype=torch.long,
            device=self.device,
        )
        self.pinch_center_bodies = []
        for body_name in self.cfg.object_pinch_center_body_names:
            self.pinch_center_bodies.append(self.hand.body_names.index(body_name))
        self.contact_sensor_ids = []
        for body_name in self.cfg.fingertip_body_names:
            sensor_ids, _ = self.finger_contact_sensor.find_sensors(body_name)
            if len(sensor_ids) == 0:
                raise RuntimeError(f"Contact sensor body not found: {body_name}")
            self.contact_sensor_ids.append(sensor_ids[0])

        joint_pos_limits = wp.to_torch(self.hand.data.joint_limits).to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]
        self.hand_init_usd_dof_pos = self._load_hand_init_usd_dof_pos()
        self.hand_init_usd_root_pose = self._load_hand_init_usd_root_pose()
        self.hand_init_usd_object_pose = self._load_hand_init_usd_object_pose()
        self.hand_init_usd_world_offset = torch.tensor(
            self.cfg.hand_init_usd_world_offset, dtype=torch.float32, device=self.device
        )

        self.actions = torch.zeros((self.num_envs, len(self.actuated_dof_indices)), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.obs_history = torch.zeros(
            (self.num_envs, self.cfg.history_len, len(self.actuated_dof_indices) * 2),
            dtype=torch.float,
            device=self.device,
        )

        self.object_init_pos = torch.tensor(self.cfg.object_init_pos, dtype=torch.float, device=self.device)
        self.object_pinch_center_offset = torch.tensor(
            self.cfg.object_pinch_center_offset, dtype=torch.float, device=self.device
        )
        self._pinch_center_object_init_pos_ready = False
        target_axis = torch.tensor(self.cfg.target_axis, dtype=torch.float, device=self.device)
        self.target_axis = target_axis / torch.clamp(torch.linalg.norm(target_axis), min=1.0e-6)

        self.reset_hand_dof_pos = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.joint_delta_sum_100 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.joint_delta_count_100 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.rotation_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_saturation_rate = torch.zeros((), dtype=torch.float, device=self.device)
        self.terminal_episode_count = torch.zeros((), dtype=torch.float, device=self.device)
        self.terminal_rotation_count = torch.zeros((), dtype=torch.float, device=self.device)
        self.terminal_success_rate = torch.zeros((), dtype=torch.float, device=self.device)
        self.terminal_drop_rate = torch.zeros((), dtype=torch.float, device=self.device)
        self.terminal_timeout_rate = torch.zeros((), dtype=torch.float, device=self.device)
        self.terminal_object_pos_diff = torch.zeros((), dtype=torch.float, device=self.device)
        self.object_pos_prev = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_default_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.reset_height_lower = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_height_upper = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        scale_num = int(self.cfg.scale_range[2])
        if scale_num <= 0:
            raise ValueError(f"Invalid scale_range[2]={self.cfg.scale_range[2]}; expected a positive integer.")
        if self.num_envs % scale_num != 0:
            raise ValueError(f"num_envs={self.num_envs} must be divisible by scale_range[2]={scale_num}.")
        self.scale_num = scale_num
        self.scale_ids = torch.arange(scale_num, device=self.device, dtype=torch.long).repeat_interleave(
            self.num_envs // scale_num
        )
        self.bucket_env = self.num_envs // scale_num
        self.bucket_grasp = 0
        self.grasp_cache = self._load_grasp_cache()
        self._logged_cache_reset = False
        self._intermediate_values_step = -1

        self._compute_intermediate_values()
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.finger_contact_sensor = ContactSensor(self.cfg.contact_sensor)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        clone_plan = cloner.ClonePlan.from_env_0(
            "/World/envs/env_0",
            "/World/envs/env_{}",
            self.scene.num_envs,
            self.scene.device,
            positions=self.scene.env_origins,
        )
        cloner.replicate(clone_plan, stage=self.scene.stage)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["finger_contact"] = self.finger_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.object_pos_prev[:] = self.object_pos
        self.object_rot_prev[:] = self.object_rot
        self.actions = torch.clamp(actions, -1.0, 1.0)
        targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.action_scale * self.actions
        lower_limits = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper_limits = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets,
            lower_limits,
            upper_limits,
        )
        self.action_saturation_rate = ((targets <= lower_limits) | (targets >= upper_limits)).float().mean()

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target_index(
            target=self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        current_obs = torch.cat(
            (
                unscale(
                    self.hand_dof_pos[:, self.actuated_dof_indices],
                    self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[:, self.actuated_dof_indices],
                ),
                unscale(
                    self.prev_targets[:, self.actuated_dof_indices],
                    self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[:, self.actuated_dof_indices],
                ),
            ),
            dim=-1,
        )
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = current_obs
        sensor_obs = torch.cat(
            (
                self.cfg.fingertip_rel_pos_obs_scale
                * (self.fingertip_pos - self.object_pos.unsqueeze(1)).reshape(self.num_envs, -1),
                self.cfg.contact_obs_force_scale * torch.clamp(self.fingertip_contact_force, max=10.0),
                self.object_pos - self.object_default_pose[:, :3],
                self.cfg.object_linvel_obs_scale * self.object_linvel,
                self.cfg.object_angvel_obs_scale * self.object_angvel,
            ),
            dim=-1,
        )
        return {"policy": torch.cat((self.obs_history.reshape(self.num_envs, -1), sensor_obs), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        object_angvel = axis_angle_from_quat(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))) / self.step_dt
        axis_angvel = torch.sum(object_angvel * self.target_axis, dim=-1)
        raw_rotate_reward = torch.clamp(axis_angvel, min=self.cfg.angvel_clip_min, max=self.cfg.angvel_clip_max)
        rotate_reward = raw_rotate_reward

        diagnostic_log: dict[str, torch.Tensor] | None = None
        if self.cfg.enable_diagnostics:
            off_axis_penalty, diagnostic_log = self._compute_diagnostics(
                object_angvel, axis_angvel, raw_rotate_reward
            )
        elif self.cfg.off_axis_angvel_penalty_scale != 0.0:
            off_axis_penalty = self._compute_off_axis_penalty(object_angvel, axis_angvel)
        else:
            off_axis_penalty = torch.zeros_like(axis_angvel)
        object_linvel_penalty = torch.norm(self.object_pos - self.object_pos_prev, p=1, dim=-1) / self.step_dt
        pos_diff_penalty = (
            (
                self.hand_dof_pos[:, self.actuated_dof_indices]
                - self.reset_hand_dof_pos[:, self.actuated_dof_indices]
            )
            ** 2
        ).sum(dim=-1)
        torque_penalty = (self.hand_dof_torque[:, self.actuated_dof_indices] ** 2).sum(dim=-1)
        work_penalty = (
            (
                self.hand_dof_torque[:, self.actuated_dof_indices]
                * self.hand_dof_vel[:, self.actuated_dof_indices]
            ).sum(dim=-1)
        ) ** 2
        object_pos_error = torch.linalg.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        object_pos_reward = 1.0 / (object_pos_error + 0.001)

        reward = compute_rewards(
            rotate_reward,
            self.cfg.rotate_reward_scale,
            object_linvel_penalty,
            self.cfg.object_linvel_penalty_scale,
            off_axis_penalty,
            self.cfg.off_axis_angvel_penalty_scale,
            pos_diff_penalty,
            self.cfg.pos_diff_penalty_scale,
            torque_penalty,
            self.cfg.torque_penalty_scale,
            work_penalty,
            self.cfg.work_penalty_scale,
            object_pos_reward,
            self.cfg.object_pos_reward_scale,
        )

        self.rotation_count += axis_angvel * self.step_dt / (2.0 * math.pi)

        self.extras["log"] = {
            "rotate/ang_vel": axis_angvel.mean(),
            "rotate/rotate_reward": rotate_reward.mean(),
            "rotate/raw_rotate_reward": raw_rotate_reward.mean(),
            "rotate/positive_vel_ratio": (axis_angvel > 0.0).float().mean(),
            "rotate/reverse_ratio": (axis_angvel < 0.0).float().mean(),
            "rotate/rotation_count": self.rotation_count.mean(),
            "rotate/drop_rate": self.last_drop.float().mean(),
            "rotate/success_rate": (self.rotation_count > self.cfg.success_rotation_count).float().mean(),
            "rotate/action_abs_mean": self.actions.abs().mean(),
            "rotate/action_saturation_rate": self.action_saturation_rate,
            "rotate/terminal_episode_count": self.terminal_episode_count,
            "rotate/terminal_rotation_count": self.terminal_rotation_count,
            "rotate/terminal_success_rate": self.terminal_success_rate,
            "rotate/terminal_drop_rate": self.terminal_drop_rate,
            "rotate/terminal_timeout_rate": self.terminal_timeout_rate,
            "rotate/terminal_object_pos_diff": self.terminal_object_pos_diff,
            "rotate/object_linvel_penalty": object_linvel_penalty.mean(),
            "rotate/pos_diff_penalty": pos_diff_penalty.mean(),
            "rotate/off_axis_penalty": off_axis_penalty.mean(),
            "rotate/torque_penalty": torque_penalty.mean(),
            "rotate/work_penalty": work_penalty.mean(),
            "rotate/object_pos_reward": object_pos_reward.mean(),
            "rotate/roll": object_angvel[:, 0].mean(),
            "rotate/pitch": object_angvel[:, 1].mean(),
            "rotate/yaw": object_angvel[:, 2].mean(),
            "rotate/total_reward": reward.mean(),
            "rotate/object_pos_diff": object_pos_error.mean(),
            "rotate/object_linvel": torch.linalg.norm(self.object_pos - self.object_pos_prev, dim=-1).mean()
            / self.step_dt,
            "rotate/object_x": self.object_pos[:, 0].mean(),
            "rotate/object_y": self.object_pos[:, 1].mean(),
            "rotate/object_z": self.object_pos[:, 2].mean(),
            "rotate/mean_episode_length": self.episode_length_buf.float().mean(),
            "rotate/gravity_z": self.physics_sim_view.get_gravity()[2],
        }
        if diagnostic_log is not None:
            self.extras["log"].update(diagnostic_log)

        return reward

    def _compute_off_axis_penalty(self, object_angvel: torch.Tensor, axis_angvel: torch.Tensor) -> torch.Tensor:
        """Compute the optional off-axis reward term without diagnostic logging."""
        fingertip_rel_pos = self.fingertip_pos - self.object_pos.unsqueeze(1)
        fingertip_dist = torch.linalg.norm(fingertip_rel_pos, dim=-1)
        distance_gate = self._distance_gate(fingertip_dist)
        force_gate = self._force_gate(self.fingertip_contact_force)
        side_wall_gate = torch.exp(-torch.abs(fingertip_rel_pos[..., 2]) / self.cfg.side_wall_z_std)
        finger_quality = distance_gate * force_gate * side_wall_gate
        non_thumb_support = finger_quality[:, self.non_thumb_finger_ids].min(dim=-1).values
        thumb_support = finger_quality[:, self.thumb_finger_id]
        rotate_support_quality = 0.5 * (thumb_support + non_thumb_support)
        off_axis_angvel = torch.linalg.norm(
            object_angvel - axis_angvel.unsqueeze(-1) * self.target_axis.unsqueeze(0), dim=-1
        )
        return off_axis_angvel * rotate_support_quality

    def _compute_diagnostics(
        self, object_angvel: torch.Tensor, axis_angvel: torch.Tensor, raw_rotate_reward: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return optional contact and grasp diagnostics for debugging training."""
        fingertip_rel_pos = self.fingertip_pos - self.object_pos.unsqueeze(1)
        fingertip_dist = torch.linalg.norm(fingertip_rel_pos, dim=-1)
        closest_fingertip_dist = torch.topk(fingertip_dist, k=2, dim=-1, largest=False).values.mean(dim=-1)
        mean_fingertip_dist = fingertip_dist.mean(dim=-1)
        thumb_dist = fingertip_dist[:, self.thumb_finger_id]
        distance_gate = self._distance_gate(fingertip_dist)
        force_gate_per_finger = self._force_gate(self.fingertip_contact_force)
        side_wall_gate = torch.exp(-torch.abs(fingertip_rel_pos[..., 2]) / self.cfg.side_wall_z_std)
        finger_quality = distance_gate * force_gate_per_finger * side_wall_gate

        top2_quality = torch.topk(finger_quality, k=2, dim=-1, largest=True).values
        top3_quality = torch.topk(finger_quality, k=min(3, self.num_fingertips), dim=-1, largest=True).values
        rolling_contact_gate = top2_quality.min(dim=-1).values
        three_finger_quality = top3_quality.min(dim=-1).values
        non_thumb_quality = finger_quality[:, self.non_thumb_finger_ids]
        top2_non_thumb_support = torch.topk(
            non_thumb_quality, k=min(2, non_thumb_quality.shape[-1]), dim=-1, largest=True
        ).values.min(dim=-1).values
        non_thumb_support = non_thumb_quality.min(dim=-1).values
        non_thumb_support_mean = non_thumb_quality.mean(dim=-1)
        thumb_support = finger_quality[:, self.thumb_finger_id]
        ring_support = finger_quality[:, self.ring_finger_id]
        ring_proximity_reward = 1.0 - torch.tanh(fingertip_dist[:, self.ring_finger_id] / self.cfg.proximity_std)
        four_finger_quality = torch.minimum(thumb_support, non_thumb_support)
        top2_distance_gate = torch.topk(distance_gate, k=2, dim=-1, largest=True).values.mean(dim=-1)
        top2_force_gate = torch.topk(force_gate_per_finger, k=2, dim=-1, largest=True).values.mean(dim=-1)
        top2_side_gate = torch.topk(side_wall_gate, k=2, dim=-1, largest=True).values.mean(dim=-1)
        top2_abs_z = torch.topk(torch.abs(fingertip_rel_pos[..., 2]), k=2, dim=-1, largest=False).values.mean(
            dim=-1
        )
        proximity_reward = torch.topk(
            1.0 - torch.tanh(fingertip_dist / self.cfg.proximity_std), k=2, dim=-1
        ).values.mean(dim=-1)
        contact_gate = top2_distance_gate * top2_side_gate
        top2_contact_force = torch.topk(self.fingertip_contact_force, k=2, dim=-1, largest=True).values.mean(dim=-1)
        contact_count = (self.fingertip_contact_force > self.cfg.contact_force_threshold).sum(dim=-1)
        contact_count_reward = torch.clamp(contact_count.float() / float(self.cfg.min_train_contact_count), max=1.0)
        thumb_force = self.fingertip_contact_force[:, self.thumb_finger_id]

        pinch_center_pos = self._compute_pinch_center_pos()
        pinch_center_dist = torch.linalg.norm(
            self.object_pos - (pinch_center_pos + self.object_pinch_center_offset), dim=-1
        )
        pinch_center_reward = torch.exp(-pinch_center_dist / self.cfg.pinch_center_reward_std)
        under_contact_penalty = torch.clamp(float(self.cfg.min_train_contact_count) - contact_count.float(), min=0.0)
        thumb_escape_penalty = torch.clamp(
            (thumb_dist - self.cfg.thumb_escape_dist) / self.cfg.thumb_escape_width, min=0.0, max=1.0
        )

        positive_rotate_reward = torch.clamp(raw_rotate_reward, min=0.0)
        negative_rotate_reward = torch.clamp(raw_rotate_reward, max=0.0)
        thumb_opposed_support = torch.minimum(thumb_support, non_thumb_support)
        rotate_support_quality = 0.5 * (thumb_support + non_thumb_support)
        rotate_support_gate = self.cfg.rotate_support_gate_floor + (
            1.0 - self.cfg.rotate_support_gate_floor
        ) * rotate_support_quality
        support_gated_rotate_reward = negative_rotate_reward + positive_rotate_reward * rotate_support_gate
        off_axis_angvel = torch.linalg.norm(
            object_angvel - axis_angvel.unsqueeze(-1) * self.target_axis.unsqueeze(0), dim=-1
        )
        off_axis_penalty = off_axis_angvel * rotate_support_quality

        log = {
            "rotate/support_gated_rotate_reward": support_gated_rotate_reward.mean(),
            "rotate/rotate_support_gate": rotate_support_gate.mean(),
            "rotate/rotate_support_quality": rotate_support_quality.mean(),
            "rotate/off_axis_angvel": off_axis_angvel.mean(),
            "rotate/proximity_reward": proximity_reward.mean(),
            "rotate/contact_gate": contact_gate.mean(),
            "rotate/force_gate": top2_force_gate.mean(),
            "rotate/real_contact_gate": rolling_contact_gate.mean(),
            "rotate/rolling_contact_gate": rolling_contact_gate.mean(),
            "rotate/three_finger_quality": three_finger_quality.mean(),
            "rotate/four_finger_quality": four_finger_quality.mean(),
            "rotate/top2_non_thumb_support": top2_non_thumb_support.mean(),
            "rotate/non_thumb_support": non_thumb_support.mean(),
            "rotate/non_thumb_support_mean": non_thumb_support_mean.mean(),
            "rotate/thumb_support": thumb_support.mean(),
            "rotate/ring_support": ring_support.mean(),
            "rotate/ring_proximity_reward": ring_proximity_reward.mean(),
            "rotate/thumb_opposed_support": thumb_opposed_support.mean(),
            "rotate/pinch_center_dist": pinch_center_dist.mean(),
            "rotate/pinch_center_reward": pinch_center_reward.mean(),
            "rotate/under_contact_penalty": under_contact_penalty.mean(),
            "rotate/thumb_escape_penalty": thumb_escape_penalty.mean(),
            "rotate/top2_distance_gate": top2_distance_gate.mean(),
            "rotate/top2_force_gate": top2_force_gate.mean(),
            "rotate/top2_side_gate": top2_side_gate.mean(),
            "rotate/top2_abs_z": top2_abs_z.mean(),
            "rotate/thumb_contact_force": thumb_force.mean(),
            "rotate/two_finger_roll_contact": (rolling_contact_gate > 0.5).float().mean(),
            "rotate/thumb_dist": thumb_dist.mean(),
            "rotate/thumb_force": thumb_force.mean(),
            "rotate/top2_contact_force": top2_contact_force.mean(),
            "rotate/contact_count": contact_count.float().mean(),
            "rotate/contact_count_reward": contact_count_reward.mean(),
            "rotate/mean_fingertip_dist": mean_fingertip_dist.mean(),
            "rotate/closest_fingertip_dist": closest_fingertip_dist.mean(),
            "rotate/min_fingertip_dist": fingertip_dist.min(dim=-1).values.mean(),
        }
        for finger_id, finger_name in enumerate(self.fingertip_log_names):
            log[f"rotate/finger_dist/{finger_name}"] = fingertip_dist[:, finger_id].mean()
            log[f"rotate/finger_force/{finger_name}"] = self.fingertip_contact_force[:, finger_id].mean()
            log[f"rotate/finger_quality/{finger_name}"] = finger_quality[:, finger_id].mean()
        return off_axis_penalty, log

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_pos_diff = torch.linalg.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        too_far = object_pos_diff > self.cfg.drop_dist
        height_reset_upper = self.object_pos[:, 2] > self.reset_height_upper
        height_reset_lower = self.object_pos[:, 2] < self.reset_height_lower
        self.last_drop = too_far | height_reset_upper | height_reset_lower
        self.extras["height_reset_upper"] = height_reset_upper.float().mean()
        self.extras["height_reset_lower"] = height_reset_lower.float().mean()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminal = self.last_drop | time_out
        terminal_float = terminal.float()
        self.terminal_episode_count = terminal_float.sum()
        terminal_count = torch.clamp(self.terminal_episode_count, min=1.0)
        self.terminal_rotation_count = (self.rotation_count * terminal_float).sum() / terminal_count
        self.terminal_success_rate = (
            ((self.rotation_count > self.cfg.success_rotation_count) & terminal).float().sum() / terminal_count
        )
        self.terminal_drop_rate = (self.last_drop & terminal).float().sum() / terminal_count
        self.terminal_timeout_rate = (time_out & ~self.last_drop & terminal).float().sum() / terminal_count
        self.terminal_object_pos_diff = (object_pos_diff * terminal_float).sum() / terminal_count
        self._update_gravity_curriculum(height_reset_upper, height_reset_lower)
        return self.last_drop, time_out

    def _update_gravity_curriculum(self, height_reset_upper: torch.Tensor, height_reset_lower: torch.Tensor) -> None:
        if not self.cfg.gravity_curriculum:
            return
        if self.common_step_counter <= self.cfg.gravity_curriculum_start_step:
            return
        if height_reset_upper.float().mean() >= self.cfg.gravity_curriculum_height_reset_threshold:
            return
        if height_reset_lower.float().mean() >= self.cfg.gravity_curriculum_height_reset_threshold:
            return

        gravity = self.physics_sim_view.get_gravity()
        gravity_amp = math.sqrt(gravity[0] ** 2 + gravity[1] ** 2 + gravity[2] ** 2)
        if gravity_amp >= self.cfg.gravity_curriculum_max:
            return

        new_gravity_amp = min(
            self.cfg.gravity_curriculum_max,
            gravity_amp + self.cfg.gravity_curriculum_increment,
        )
        new_gravity = carb.Float3(0.0, 0.0, -new_gravity_amp)
        self.physics_sim_view.set_gravity(new_gravity)
        print(f"update gravity: {new_gravity}")

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        object_pose = wp.to_torch(self.object.data.default_root_pose).clone()[env_ids]
        object_vel = wp.to_torch(self.object.data.default_root_vel).clone()[env_ids]
        dof_pos = wp.to_torch(self.hand.data.default_joint_pos).clone()[env_ids]
        dof_vel = torch.zeros_like(wp.to_torch(self.hand.data.default_joint_vel).clone()[env_ids])

        object_pose_local = object_pose.clone()
        using_grasp_cache = self.grasp_cache is not None
        object_reset_pos_offset = torch.tensor(
            getattr(
                self.cfg,
                "cache_object_reset_pos_offset" if using_grasp_cache else "object_reset_pos_offset",
                (0.0, 0.0, 0.0),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        object_reset_z_offset = float(
            getattr(
                self.cfg,
                "cache_object_reset_z_offset" if using_grasp_cache else "object_reset_z_offset",
                0.0,
            )
        )
        pos_noise = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
        if not using_grasp_cache and self.hand_init_usd_dof_pos is not None:
            dof_pos[:] = self.hand_init_usd_dof_pos.unsqueeze(0)
        default_dof_pos = dof_pos.clone()
        if using_grasp_cache:
            cache_ids = self._sample_grasp_cache_ids(env_ids)
            cache_state = self.grasp_cache[cache_ids]
            dof_pos[:, : self.num_hand_dofs] = cache_state[:, : self.num_hand_dofs]
            object_pose_local[:, :7] = cache_state[:, self.num_hand_dofs : self.num_hand_dofs + 7]
            object_pose[:, :3] = object_pose_local[:, :3] + self.scene.env_origins[env_ids]
            object_pose[:, 3:7] = object_pose_local[:, 3:7]
            if not self._logged_cache_reset:
                print(f"[INFO] Allegro rotate reset sampled grasp cache: {self.grasp_cache_path}")
                self._logged_cache_reset = True
        else:
            self._ensure_pinch_center_object_init_pos(env_ids, default_dof_pos)
            pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device) * self.cfg.reset_position_noise
            if self.cfg.reset_dof_pos_noise > 0.0:
                dof_pos_noise = sample_uniform(-1.0, 1.0, dof_pos.shape, device=self.device)
                dof_pos += self.cfg.reset_dof_pos_noise * dof_pos_noise
                dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids])

        hand_root_pose = wp.to_torch(self.hand.data.default_root_pose).clone()[env_ids]
        hand_root_vel = wp.to_torch(self.hand.data.default_root_vel).clone()[env_ids]
        if self.hand_init_usd_root_pose is not None:
            hand_root_pose[:, :7] = self.hand_init_usd_root_pose.unsqueeze(0).expand(len(env_ids), -1)
            hand_root_pose[:, 0:3] += self.hand_init_usd_world_offset.unsqueeze(0)
        hand_root_pose[:, 0:3] += self.scene.env_origins[env_ids]
        hand_root_vel[:] = 0.0
        self.hand.write_root_pose_to_sim_index(root_pose=hand_root_pose, env_ids=env_ids)
        self.hand.write_root_velocity_to_sim_index(root_velocity=hand_root_vel, env_ids=env_ids)

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.hand.write_joint_velocity_to_sim_index(velocity=dof_vel, env_ids=env_ids)
        self.hand.set_joint_position_target_index(target=dof_pos, env_ids=env_ids)

        if not using_grasp_cache:
            if self.hand_init_usd_object_pose is not None:
                object_pose_local[:, :7] = self.hand_init_usd_object_pose.unsqueeze(0).expand(len(env_ids), -1)
                object_pose_local[:, 0:3] += self.hand_init_usd_world_offset.unsqueeze(0)
            elif self.cfg.object_init_from_fingertip_center:
                fingertip_center, object_init_pos = self._compute_reset_fingertip_center(env_ids)
                object_pose_local[:, 0:3] = object_init_pos + pos_noise
                self._log_reset_fingertip_center_stats(env_ids, fingertip_center, object_pose_local[:, 0:3])
            elif self.cfg.object_init_from_pinch_center:
                object_pose_local[:, 0:3] = self.object_init_pos + pos_noise
            else:
                object_pose_local[:, 0:3] = self.object_init_pos + pos_noise
            object_pose[:, :7] = object_pose_local[:, :7]
            object_pose[:, 0:3] = object_pose_local[:, 0:3] + self.scene.env_origins[env_ids]

        if object_reset_z_offset != 0.0:
            object_reset_pos_offset = object_reset_pos_offset.clone()
            object_reset_pos_offset[2] += object_reset_z_offset
        if torch.any(object_reset_pos_offset != 0.0):
            object_pose_local[:, 0:3] += object_reset_pos_offset.unsqueeze(0)
            object_pose[:, 0:3] += object_reset_pos_offset.unsqueeze(0)

        object_vel[:] = 0.0
        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(root_velocity=object_vel, env_ids=env_ids)
        if not using_grasp_cache:
            self.sim.forward()
            self.scene.update(dt=0.0)
        self.object_default_pose[env_ids, :7] = object_pose_local[:, :7]
        height_window = self.cfg.reset_height_upper - self.cfg.reset_height_lower
        self.reset_height_lower[env_ids] = object_pose_local[:, 2] - height_window / 2.0
        self.reset_height_upper[env_ids] = object_pose_local[:, 2] + height_window / 2.0

        self.rotation_count[env_ids] = 0.0
        self.last_drop[env_ids] = False
        self.reset_hand_dof_pos[env_ids] = dof_pos
        self.joint_delta_sum_100[env_ids] = 0.0
        self.joint_delta_count_100[env_ids] = 0.0

        obs = torch.cat(
            (
                unscale(
                    dof_pos[:, self.actuated_dof_indices],
                    self.hand_dof_lower_limits[env_ids][:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[env_ids][:, self.actuated_dof_indices],
                ),
                unscale(
                    dof_pos[:, self.actuated_dof_indices],
                    self.hand_dof_lower_limits[env_ids][:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[env_ids][:, self.actuated_dof_indices],
                ),
            ),
            dim=-1,
        )
        self.obs_history[env_ids] = obs.unsqueeze(1).repeat(1, self.cfg.history_len, 1)
        self._compute_intermediate_values(force=True)
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

    def _compute_reset_fingertip_center(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        fingertip_pos_w = wp.to_torch(self.hand.data.body_pos_w)[:, self.finger_bodies][env_ids]
        fingertip_pos = fingertip_pos_w - self.scene.env_origins[env_ids].unsqueeze(1)
        offset = torch.tensor(self.cfg.object_fingertip_center_offset, dtype=torch.float, device=self.device)
        fingertip_center = fingertip_pos.mean(dim=1)
        return fingertip_center, fingertip_center + offset

    def _log_reset_fingertip_center_stats(
        self, env_ids: torch.Tensor, fingertip_center: torch.Tensor, object_init_pos: torch.Tensor
    ) -> None:
        if not getattr(self.cfg, "log_reset_fingertip_center", False):
            return
        interval = max(1, int(getattr(self.cfg, "reset_fingertip_center_log_interval", 40)))
        if self.common_step_counter % interval != 0:
            return
        distance_to_center = torch.linalg.norm(object_init_pos - fingertip_center, dim=-1)
        center_mean = fingertip_center.mean(dim=0)
        center_std = fingertip_center.std(dim=0, unbiased=False)
        object_mean = object_init_pos.mean(dim=0)
        object_std = object_init_pos.std(dim=0, unbiased=False)
        print(
            "[reset_init] Allegro fingertip_center_mean="
            f"({center_mean[0].item():+.4f},{center_mean[1].item():+.4f},{center_mean[2].item():+.4f}), "
            f"fingertip_center_std=({center_std[0].item():.4f},{center_std[1].item():.4f},{center_std[2].item():.4f}), "
            f"object_init_mean=({object_mean[0].item():+.4f},{object_mean[1].item():+.4f},{object_mean[2].item():+.4f}), "
            f"object_init_std=({object_std[0].item():.4f},{object_std[1].item():.4f},{object_std[2].item():.4f}), "
            f"distance_to_center_mean={distance_to_center.mean().item():.4f}, "
            f"distance_to_center_std={distance_to_center.std(unbiased=False).item():.4f}"
        )

    def _ensure_pinch_center_object_init_pos(self, env_ids: torch.Tensor, dof_pos: torch.Tensor) -> None:
        if not self.cfg.object_init_from_pinch_center or self._pinch_center_object_init_pos_ready:
            return

        hand_root_pose = wp.to_torch(self.hand.data.default_root_pose).clone()[env_ids]
        hand_root_vel = wp.to_torch(self.hand.data.default_root_vel).clone()[env_ids]
        hand_root_pose[:, 0:3] += self.scene.env_origins[env_ids]
        hand_root_vel[:] = 0.0
        dof_vel = torch.zeros_like(wp.to_torch(self.hand.data.default_joint_vel).clone()[env_ids])

        self.hand.write_root_pose_to_sim_index(root_pose=hand_root_pose, env_ids=env_ids)
        self.hand.write_root_velocity_to_sim_index(root_velocity=hand_root_vel, env_ids=env_ids)
        self.hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.hand.write_joint_velocity_to_sim_index(velocity=dof_vel, env_ids=env_ids)
        self.hand.set_joint_position_target_index(target=dof_pos, env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

        pinch_center = self._compute_pinch_center_pos(env_ids).mean(dim=0)
        offset = torch.tensor(self.cfg.object_pinch_center_offset, dtype=torch.float, device=self.device)
        self.object_init_pos[:] = pinch_center + offset
        self._pinch_center_object_init_pos_ready = True
        print(
            "[INFO] Allegro grasp object init calibrated from pinch center: "
            f"({self.object_init_pos[0].item():+.4f},"
            f"{self.object_init_pos[1].item():+.4f},"
            f"{self.object_init_pos[2].item():+.4f})"
        )

    def _compute_pinch_center_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        pinch_pos_w = wp.to_torch(self.hand.data.body_pos_w)[:, self.pinch_center_bodies][env_ids]
        pinch_pos = pinch_pos_w - self.scene.env_origins[env_ids].unsqueeze(1)
        return pinch_pos.mean(dim=1)

    def _load_hand_init_usd_dof_pos(self) -> torch.Tensor | None:
        usd_path = getattr(self.cfg, "hand_init_usd_path", "")
        if not usd_path:
            return None
        source_prim_path = getattr(self.cfg, "hand_init_usd_prim_path", "")
        if not source_prim_path:
            raise ValueError("cfg.hand_init_usd_path is set but cfg.hand_init_usd_prim_path is empty.")
        if not os.path.isabs(usd_path):
            usd_path = os.path.abspath(usd_path)

        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(usd_path)
        if layer is None:
            raise FileNotFoundError(f"Could not open Allegro hand init USD: {usd_path}")
        src_root = layer.GetPrimAtPath(source_prim_path)
        if src_root is None:
            raise ValueError(f"Missing Allegro hand init prim {source_prim_path} in {usd_path}")

        def _prop_default(spec, name: str):
            for prop in list(getattr(spec, "properties", [])):
                if prop.name == name and "default" in prop.ListInfoKeys():
                    return prop.GetInfo("default")
            return None

        joint_positions: dict[str, float] = {}

        def _walk_joint_specs(src_spec) -> None:
            target_deg = _prop_default(src_spec, "drive:angular:physics:targetPosition")
            lower_deg = _prop_default(src_spec, "physics:lowerLimit")
            upper_deg = _prop_default(src_spec, "physics:upperLimit")
            if "_joint_" in src_spec.name and (
                target_deg is not None or lower_deg is not None or upper_deg is not None
            ):
                chosen_deg = 0.0 if target_deg is None else float(target_deg)
                if lower_deg is not None:
                    chosen_deg = max(chosen_deg, float(lower_deg))
                if upper_deg is not None:
                    chosen_deg = min(chosen_deg, float(upper_deg))
                joint_positions[src_spec.name] = math.radians(chosen_deg)
            for child in list(getattr(src_spec, "nameChildren", [])):
                _walk_joint_specs(child)

        _walk_joint_specs(src_root)
        if not joint_positions:
            raise ValueError(f"No authored Allegro joint target positions found under {source_prim_path} in {usd_path}")

        dof_pos = wp.to_torch(self.hand.data.default_joint_pos).clone()[0].to(self.device)
        matched_names: list[str] = []
        for joint_name, joint_pos in joint_positions.items():
            if joint_name not in self.hand.joint_names:
                continue
            joint_id = self.hand.joint_names.index(joint_name)
            dof_pos[joint_id] = joint_pos
            matched_names.append(joint_name)
        if not matched_names:
            raise ValueError(
                f"No USD joint names from {usd_path}:{source_prim_path} matched Allegro sim joints "
                f"{self.hand.joint_names}."
            )
        dof_pos = torch.clamp(dof_pos, self.hand_dof_lower_limits[0], self.hand_dof_upper_limits[0])
        print(
            "[INFO] Loaded Allegro hand init pose from USD: "
            f"{usd_path}, prim={source_prim_path}, joints={len(matched_names)}"
        )
        return dof_pos

    def _load_hand_init_usd_root_pose(self) -> torch.Tensor | None:
        usd_path = getattr(self.cfg, "hand_init_usd_path", "")
        source_prim_path = getattr(self.cfg, "hand_init_usd_prim_path", "")
        if not usd_path or not source_prim_path:
            return None
        if not os.path.isabs(usd_path):
            usd_path = os.path.abspath(usd_path)

        from pxr import Sdf, Usd

        layer = Sdf.Layer.FindOrOpen(usd_path)
        if layer is None:
            raise FileNotFoundError(f"Could not open Allegro hand init USD: {usd_path}")
        stage = Usd.Stage.Open(layer.identifier)
        if stage is None:
            raise FileNotFoundError(f"Could not open Allegro hand init USD stage: {usd_path}")
        hand_prim = stage.GetPrimAtPath(source_prim_path)
        if not hand_prim.IsValid():
            raise ValueError(f"Missing Allegro hand reference prim {source_prim_path} in {usd_path}")

        hand_pos_usd, hand_quat_usd = _usd_world_pose_xyzw(hand_prim)
        hand_pose = torch.tensor((*hand_pos_usd, *hand_quat_usd), dtype=torch.float32, device=self.device)
        print(
            "[INFO] Loaded Allegro hand root pose directly from USD: "
            f"{usd_path}, hand_prim={source_prim_path}, "
            f"pose=({hand_pose[0].item():+.4f},{hand_pose[1].item():+.4f},"
            f"{hand_pose[2].item():+.4f}; {hand_pose[3].item():+.4f},"
            f"{hand_pose[4].item():+.4f},{hand_pose[5].item():+.4f},"
            f"{hand_pose[6].item():+.4f})"
        )
        return hand_pose

    def _load_hand_init_usd_object_pose(self) -> torch.Tensor | None:
        usd_path = getattr(self.cfg, "hand_init_usd_path", "")
        object_prim_path = getattr(self.cfg, "hand_init_usd_object_prim_path", "")
        if not usd_path or not object_prim_path:
            return None
        if not os.path.isabs(usd_path):
            usd_path = os.path.abspath(usd_path)

        from pxr import Sdf, Usd

        layer = Sdf.Layer.FindOrOpen(usd_path)
        if layer is None:
            raise FileNotFoundError(f"Could not open Allegro hand/object init USD: {usd_path}")
        stage = Usd.Stage.Open(layer.identifier)
        if stage is None:
            raise FileNotFoundError(f"Could not open Allegro hand/object init USD stage: {usd_path}")
        object_prim = stage.GetPrimAtPath(object_prim_path)
        if not object_prim.IsValid():
            raise ValueError(f"Missing Allegro object reference prim {object_prim_path} in {usd_path}")

        object_pos_usd, object_quat_usd = _usd_world_pose_xyzw(object_prim)
        object_quat = _quat_normalize_xyzw(tuple(float(value) for value in self.cfg.object_init_rot))
        object_pose = torch.tensor((*object_pos_usd, *object_quat), dtype=torch.float32, device=self.device)
        object_z_axis = _quat_apply_xyzw(object_quat_usd, (0.0, 0.0, 1.0))
        object_z_norm = math.sqrt(sum(value * value for value in object_z_axis))
        object_z_cos = object_z_axis[2] / max(object_z_norm, 1.0e-12)
        usd_object_z_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, object_z_cos))))
        forced_object_z_axis = _quat_apply_xyzw(object_quat, (0.0, 0.0, 1.0))
        forced_object_z_norm = math.sqrt(sum(value * value for value in forced_object_z_axis))
        forced_object_z_cos = forced_object_z_axis[2] / max(forced_object_z_norm, 1.0e-12)
        forced_object_z_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, forced_object_z_cos))))
        print(
            "[INFO] Loaded Allegro object init position from USD with configured upright rotation: "
            f"{usd_path}, object_prim={object_prim_path}, "
            f"pose=({object_pose[0].item():+.4f},{object_pose[1].item():+.4f},"
            f"{object_pose[2].item():+.4f}; {object_pose[3].item():+.4f},"
            f"{object_pose[4].item():+.4f},{object_pose[5].item():+.4f},"
            f"{object_pose[6].item():+.4f}), "
            f"usd_z_axis_angle={usd_object_z_angle_deg:.2f}deg, "
            f"forced_z_axis_angle={forced_object_z_angle_deg:.2f}deg"
        )
        return object_pose

    def _load_grasp_cache(self) -> torch.Tensor | None:
        self.grasp_cache_path = ""
        require_grasp_cache = getattr(self.cfg, "require_grasp_cache", False)
        if not self.cfg.grasp_cache_path:
            if require_grasp_cache:
                raise ValueError("Allegro rotate requires cfg.grasp_cache_path, but it is empty.")
            return None

        cache_path = allegro_grasp_cache_path(self.cfg.grasp_cache_path, self.cfg.scale_range)
        if not os.path.isabs(cache_path):
            cache_path = os.path.abspath(cache_path)
        self.grasp_cache_path = cache_path
        if not os.path.exists(cache_path):
            if require_grasp_cache:
                raise FileNotFoundError(
                    "No saved Allegro grasping states found. Generate the cache first with "
                    "`source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/tools/allegro_gen_grasp.py`. "
                    f"Expected path: {cache_path}"
                )
            return None

        cache = np.load(cache_path)
        expected_dim = self.num_hand_dofs + 7
        if cache.ndim != 2 or cache.shape[1] != expected_dim:
            raise ValueError(
                f"Invalid Allegro roll grasp cache shape {cache.shape}; expected (N, {expected_dim}) "
                f"for {self.num_hand_dofs} hand DOFs plus object pose."
            )
        cache_tensor = torch.as_tensor(cache, dtype=torch.float32, device=self.device)
        if cache_tensor.shape[0] % self.scale_num != 0:
            raise ValueError(
                f"Invalid Allegro grasp cache row count {cache_tensor.shape[0]}; "
                f"expected divisible by scale_range[2]={self.scale_num}."
            )
        self.bucket_grasp = cache_tensor.shape[0] // self.scale_num
        print(f"[INFO] Loaded Allegro grasp cache: {cache_path}, shape={tuple(cache_tensor.shape)}")
        return cache_tensor

    def zero_actions(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, len(self.actuated_dof_indices)), dtype=torch.float, device=self.device)

    def _sample_grasp_cache_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        cache_ids = torch.empty((len(env_ids),), dtype=torch.long, device=self.device)
        env_scale_ids = self.scale_ids[env_ids]
        for scale_id in range(self.scale_num):
            mask = env_scale_ids == scale_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            local_ids = torch.randint(0, self.bucket_grasp, (count,), device=self.device)
            cache_ids[mask] = scale_id * self.bucket_grasp + local_ids
        return cache_ids

    def _compute_intermediate_values(self, force: bool = False):
        if not force and self._intermediate_values_step == self.common_step_counter:
            return
        self.fingertip_pos = wp.to_torch(self.hand.data.body_pos_w)[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )

        self.hand_dof_pos = wp.to_torch(self.hand.data.joint_pos)
        self.hand_dof_vel = wp.to_torch(self.hand.data.joint_vel)
        self.hand_dof_torque = wp.to_torch(self.hand.data.applied_torque)

        self.object_pos = wp.to_torch(self.object.data.root_pos_w) - self.scene.env_origins
        self.object_rot = wp.to_torch(self.object.data.root_quat_w)
        self.object_linvel = wp.to_torch(self.object.data.root_lin_vel_w)
        self.object_angvel = wp.to_torch(self.object.data.root_ang_vel_w)

        net_contact_forces = wp.to_torch(self.finger_contact_sensor.data.net_forces_w_history)
        self.fingertip_contact_force = torch.max(
            torch.linalg.norm(net_contact_forces[:, :, self.contact_sensor_ids], dim=-1),
            dim=1,
        )[0]
        self._intermediate_values_step = self.common_step_counter

    def _distance_gate(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (self.cfg.contact_gate_dist - dist) / self.cfg.contact_gate_width,
            min=0.0,
            max=1.0,
        )

    def _force_gate(self, force: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (force - self.cfg.contact_force_threshold) / self.cfg.contact_force_width,
            min=0.0,
            max=1.0,
        )


def _usd_world_pose_xyzw(prim) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quat_wxyz = matrix.ExtractRotationQuat()
    imaginary = quat_wxyz.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        _quat_normalize_xyzw(
            (float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(quat_wxyz.GetReal()))
        ),
    )


def _quat_normalize_xyzw(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in quat)


def _quat_conjugate_xyzw(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-quat[0], -quat[1], -quat[2], quat[3])


def _quat_raw_mul_xyzw(
    quat_1: tuple[float, float, float, float], quat_2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = quat_1
    x2, y2, z2, w2 = quat_2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_apply_xyzw(
    quat: tuple[float, float, float, float], vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    quat = _quat_normalize_xyzw(quat)
    rotated = _quat_raw_mul_xyzw(_quat_raw_mul_xyzw(quat, (vec[0], vec[1], vec[2], 0.0)), _quat_conjugate_xyzw(quat))
    return (rotated[0], rotated[1], rotated[2])


def _vec_add(vec_1: tuple[float, float, float], vec_2: tuple[float, float, float]) -> tuple[float, float, float]:
    return (vec_1[0] + vec_2[0], vec_1[1] + vec_2[1], vec_1[2] + vec_2[2])


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def compute_rewards(
    rotate_reward: torch.Tensor,
    rotate_reward_scale: float,
    object_linvel_penalty: torch.Tensor,
    object_linvel_penalty_scale: float,
    off_axis_penalty: torch.Tensor,
    off_axis_angvel_penalty_scale: float,
    pos_diff_penalty: torch.Tensor,
    pos_diff_penalty_scale: float,
    torque_penalty: torch.Tensor,
    torque_penalty_scale: float,
    work_penalty: torch.Tensor,
    work_penalty_scale: float,
    object_pos_reward: torch.Tensor,
    object_pos_reward_scale: float,
) -> torch.Tensor:
    reward = rotate_reward * rotate_reward_scale
    reward += object_linvel_penalty * object_linvel_penalty_scale
    reward += off_axis_penalty * off_axis_angvel_penalty_scale
    reward += pos_diff_penalty * pos_diff_penalty_scale
    reward += torque_penalty * torque_penalty_scale
    reward += work_penalty * work_penalty_scale
    reward += object_pos_reward * object_pos_reward_scale
    return reward
