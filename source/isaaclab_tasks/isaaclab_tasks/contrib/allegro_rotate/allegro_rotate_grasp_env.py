# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import carb

import isaaclab.sim as sim_utils
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from .allegro_rotate_env import AllegroRotateEnv
from .allegro_rotate_env_cfg import allegro_grasp_cache_path
from .allegro_rotate_grasp_env_cfg import AllegroRotateGraspEnvCfg


class AllegroRotateGraspEnv(AllegroRotateEnv):
    """Generate stable Allegro grasp states for cache-based rotation training."""

    cfg: AllegroRotateGraspEnvCfg

    def __init__(self, cfg: AllegroRotateGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self._grasp_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        self._validate_static_sampling_cfg()
        self._probe_drop_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        scale_num = int(self.cfg.scale_range[2])
        if scale_num <= 0:
            raise ValueError(f"Invalid scale_range[2]={self.cfg.scale_range[2]}; expected a positive integer.")

        self.grasp_cache_target_per_scale = int(self.cfg.grasp_cache_target // scale_num)
        if self.grasp_cache_target_per_scale <= 0:
            raise ValueError(
                f"Invalid grasp_cache_target={self.cfg.grasp_cache_target}; "
                f"expected at least scale_range[2]={scale_num}."
            )
        self.saved_grasping_states = [
            torch.zeros((0, self.num_hand_dofs + 7), dtype=torch.float32, device=self.device) for _ in range(scale_num)
        ]
        self.grasp_contact_sensor_ids = []
        grasp_contact_finger_ids = []
        for body_name in self.cfg.grasp_contact_body_names:
            sensor_ids, _ = self.finger_contact_sensor.find_sensors(body_name)
            if len(sensor_ids) == 0:
                raise RuntimeError(f"Grasp contact sensor body not found: {body_name}")
            self.grasp_contact_sensor_ids.append(sensor_ids[0])
            finger_name = body_name.split("_link_", 1)[0]
            grasp_contact_finger_ids.append(self.fingertip_log_names.index(finger_name))
        self.grasp_contact_finger_ids = torch.tensor(grasp_contact_finger_ids, dtype=torch.long, device=self.device)
        self._last_grasp_stable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_grasp_metrics: dict[str, torch.Tensor] = {}
        self.gravity_id = 0
        self.gravity_all_directions = [
            carb.Float3(0.0, 0.0, 9.81),
            carb.Float3(0.0, 0.0, -9.81),
            carb.Float3(0.0, 9.81, 0.0),
            carb.Float3(0.0, -9.81, 0.0),
            carb.Float3(9.81, 0.0, 0.0),
            carb.Float3(-9.81, 0.0, 0.0),
        ]
        self.physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        self._log_grasp_physics_cfg()
        self._grasp_ready = True

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        stable, metrics = self._compute_grasp_success()
        self._last_grasp_stable = stable
        self._last_grasp_metrics = metrics
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.grasp_probe_init_pose:
            object_dropped = (metrics["object_pos_diff"] > self.cfg.grasp_max_object_pos_diff) | ~metrics["height_ok"]
            probe_failed = object_dropped | ~stable
            if not hasattr(self, "_probe_drop_counter"):
                self._probe_drop_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            if self.cfg.grasp_probe_drop_hold_steps > 0:
                self._probe_drop_counter = torch.where(
                    probe_failed,
                    self._probe_drop_counter + 1,
                    torch.zeros_like(self._probe_drop_counter),
                )
                probe_terminated = self._probe_drop_counter >= self.cfg.grasp_probe_drop_hold_steps
            else:
                self._probe_drop_counter[:] = 0
                probe_terminated = probe_failed
            metrics["probe_drop"] = object_dropped
            metrics["probe_failed"] = probe_failed
            metrics["probe_drop_counter"] = self._probe_drop_counter
            return probe_terminated, time_out
        terminated = ~stable
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        stable = self._last_grasp_stable
        metrics = self._last_grasp_metrics
        if (
            self.cfg.gravity_curriculum
            and not self.cfg.grasp_probe_init_pose
            and self.cfg.grasp_gravity_switch_interval > 0
            and self.common_step_counter % self.cfg.grasp_gravity_switch_interval == 0
        ):
            self.physics_sim_view.set_gravity(self.gravity_all_directions[self.gravity_id])
            self.gravity_id = (self.gravity_id + 1) % len(self.gravity_all_directions)

        if metrics:
            self.extras["log"] = {
                "grasp/stable_rate": stable.float().mean(),
                "grasp/contact_count": metrics["contact_count"].float().mean(),
                "grasp/contact_finger_count": metrics["contact_finger_count"].float().mean(),
                "grasp/thumb_contact_rate": metrics["thumb_contact_ok"].float().mean(),
                "grasp/fingertip_contact_count": metrics["fingertip_contact_count"].float().mean(),
                "grasp/near_count": metrics["near_count"].float().mean(),
                "grasp/min_fingertip_dist": metrics["min_fingertip_dist"].mean(),
                "grasp/pinch_center_dist": metrics["pinch_center_dist"].mean(),
                "grasp/joint_delta_100_deg": metrics["joint_delta_100_deg"].mean(),
                "grasp/target_equal_error_deg": metrics["target_equal_error_deg"].mean(),
                "grasp/object_pos_diff": metrics["object_pos_diff"].mean(),
                "grasp/object_rot_diff": metrics["object_rot_diff"].mean(),
            }
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if getattr(self, "_grasp_ready", False):
            env_ids_tensor = self._normalize_env_ids(env_ids)
            self._save_successful_grasps(env_ids_tensor)
        super()._reset_idx(env_ids)
        if hasattr(self, "_probe_drop_counter"):
            env_ids_tensor = self._normalize_env_ids(env_ids)
            self._probe_drop_counter[env_ids_tensor] = 0

    def _compute_grasp_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._compute_intermediate_values()
        fingertip_rel_pos = self.fingertip_pos - self.object_pos.unsqueeze(1)
        fingertip_dist = torch.linalg.norm(fingertip_rel_pos, dim=-1)
        pinch_center_pos = self._compute_pinch_center_pos()
        pinch_center_offset = torch.tensor(self.cfg.object_pinch_center_offset, dtype=torch.float, device=self.device)
        target_pinch_center_pos = pinch_center_pos + pinch_center_offset
        pinch_center_dist = torch.linalg.norm(self.object_pos - target_pinch_center_pos, dim=-1)
        net_contact_forces = torch.max(
            torch.linalg.norm(self._grasp_contact_forces(), dim=-1),
            dim=1,
        )[0]
        contact_count = (net_contact_forces > self.cfg.grasp_contact_force_threshold).sum(dim=-1)
        finger_contact_force = torch.zeros(
            (self.num_envs, self.num_fingertips), dtype=net_contact_forces.dtype, device=self.device
        )
        for finger_id in range(self.num_fingertips):
            finger_mask = self.grasp_contact_finger_ids == finger_id
            if torch.any(finger_mask):
                finger_contact_force[:, finger_id] = net_contact_forces[:, finger_mask].max(dim=-1).values
        contact_finger_count = (finger_contact_force > self.cfg.grasp_contact_force_threshold).sum(dim=-1)
        thumb_contact_ok = finger_contact_force[:, self.thumb_finger_id] > self.cfg.grasp_contact_force_threshold
        fingertip_contact_count = (self.fingertip_contact_force > self.cfg.grasp_contact_force_threshold).sum(dim=-1)
        near_count = (fingertip_dist < self.cfg.grasp_fingertip_dist_threshold).sum(dim=-1)
        object_pos_diff = torch.linalg.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        object_rot_diff = quat_angle_diff(self.object_rot, self.object_default_pose[:, 3:7])
        joint_delta_rad = torch.abs(
            self.hand_dof_pos[:, self.actuated_dof_indices] - self.reset_hand_dof_pos[:, self.actuated_dof_indices]
        ).mean(dim=-1)
        jitter_mask = self.episode_length_buf <= self.cfg.joint_jitter_window_steps
        self.joint_delta_sum_100[jitter_mask] += joint_delta_rad[jitter_mask]
        self.joint_delta_count_100[jitter_mask] += 1.0
        joint_delta_100_rad = self.joint_delta_sum_100 / torch.clamp(self.joint_delta_count_100, min=1.0)
        target_equal_error_rad = (
            torch.abs(self.prev_targets[:, self.actuated_dof_indices] - self.cur_targets[:, self.actuated_dof_indices])
            .max(dim=-1)
            .values
        )
        target_track_error_rad = torch.abs(
            self.hand_dof_pos[:, self.actuated_dof_indices] - self.cur_targets[:, self.actuated_dof_indices]
        ).mean(dim=-1)
        height_ok = (self.object_pos[:, 2] > self.reset_height_lower) & (
            self.object_pos[:, 2] < self.reset_height_upper
        )

        stable = (
            (near_count >= self.cfg.grasp_min_near_fingers)
            & (contact_count >= self.cfg.grasp_min_contact_bodies)
            & (contact_finger_count >= self.cfg.grasp_min_contact_fingers)
            & (pinch_center_dist < self.cfg.grasp_max_pinch_center_dist)
            & (object_pos_diff < self.cfg.grasp_max_object_pos_diff)
            & (object_rot_diff < self.cfg.grasp_reset_angle_diff)
            & height_ok
        )
        if self.cfg.grasp_require_thumb_contact:
            stable = stable & thumb_contact_ok
        metrics = {
            "contact_count": contact_count,
            "contact_finger_count": contact_finger_count,
            "finger_contact_force": finger_contact_force,
            "thumb_contact_ok": thumb_contact_ok,
            "fingertip_contact_count": fingertip_contact_count,
            "near_count": near_count,
            "min_fingertip_dist": fingertip_dist.min(dim=-1).values,
            "fingertip_dist": fingertip_dist,
            "fingertip_rel_pos": fingertip_rel_pos,
            "pinch_center_dist": pinch_center_dist,
            "pinch_center_ok": pinch_center_dist < self.cfg.pinch_center_metric_threshold,
            "fingertip_contact_force": self.fingertip_contact_force,
            "joint_delta_100_deg": joint_delta_100_rad * (180.0 / torch.pi),
            "joint_jitter_ok": joint_delta_100_rad * (180.0 / torch.pi) < self.cfg.joint_jitter_warn_threshold_deg,
            "target_equal_error_deg": target_equal_error_rad * (180.0 / torch.pi),
            "target_track_error_deg": target_track_error_rad * (180.0 / torch.pi),
            "object_pos_diff": object_pos_diff,
            "object_rot_diff": object_rot_diff,
            "height_ok": height_ok,
            "object_x": self.object_pos[:, 0],
            "object_y": self.object_pos[:, 1],
            "object_z": self.object_pos[:, 2],
        }
        return stable, metrics

    def _save_successful_grasps(self, env_ids: torch.Tensor) -> None:
        if self.cfg.grasp_probe_init_pose:
            self._print_cache_status()
            return
        if env_ids.numel() == 0:
            return

        success = self._last_grasp_stable[env_ids] & (self.episode_length_buf[env_ids] >= self.max_episode_length - 1)
        if not torch.any(success):
            self._print_cache_status()
            return

        states = torch.cat(
            (
                self.hand_dof_pos[env_ids][success],
                self.object_pos[env_ids][success],
                self.object_rot[env_ids][success],
            ),
            dim=1,
        )
        saved_scale_ids = self.scale_ids[env_ids][success]

        for state, scale_id in zip(states, saved_scale_ids):
            scale_idx = int(scale_id.item())
            current = self.saved_grasping_states[scale_idx]
            if current.shape[0] < self.grasp_cache_target_per_scale:
                self.saved_grasping_states[scale_idx] = torch.cat((current, state.reshape(1, -1)), dim=0)

        self._print_cache_status(force=True)
        if self._is_cache_complete():
            self._write_cache_and_exit()

    def _print_cache_status(self, force: bool = False) -> None:
        if not force and self.common_step_counter % self.cfg.grasp_cache_status_interval != 0:
            return
        total = sum(saved.shape[0] for saved in self.saved_grasping_states)
        finished = sum(saved.shape[0] >= self.grasp_cache_target_per_scale for saved in self.saved_grasping_states)
        suffix = ""
        if self._last_grasp_metrics:
            metrics = self._last_grasp_metrics
            suffix = (
                f", stable={self._last_grasp_stable.float().mean().item():.3f}"
                f", contacts={metrics['contact_count'].float().mean().item():.3f}"
                f", contact_fingers={metrics['contact_finger_count'].float().mean().item():.3f}"
                f", thumb_contact={metrics['thumb_contact_ok'].float().mean().item():.3f}"
                f", tip_contacts={metrics['fingertip_contact_count'].float().mean().item():.3f}"
                f", near={metrics['near_count'].float().mean().item():.3f}"
                f", min_dist={metrics['min_fingertip_dist'].mean().item():.4f}"
                f", pinch_dist={metrics['pinch_center_dist'].mean().item():.4f}"
                f", pinch_ok={metrics['pinch_center_ok'].float().mean().item():.3f}"
                f", joint_delta_100_deg={metrics['joint_delta_100_deg'].mean().item():.3f}"
                f", joint_static_ok={metrics['joint_jitter_ok'].float().mean().item():.3f}"
                f", target_eq_deg={metrics['target_equal_error_deg'].mean().item():.4f}"
                f", target_track_deg={metrics['target_track_error_deg'].mean().item():.3f}"
                f", pos_diff={metrics['object_pos_diff'].mean().item():.4f}"
                f", rot_diff={metrics['object_rot_diff'].mean().item():.4f}"
                f", height_ok={metrics['height_ok'].float().mean().item():.3f}"
                f", object=({metrics['object_x'].mean().item():+.4f},"
                f"{metrics['object_y'].mean().item():+.4f},"
                f"{metrics['object_z'].mean().item():+.4f})"
            )
            if "probe_drop_counter" in metrics:
                suffix += (
                    f", probe_drop={metrics['probe_drop'].float().mean().item():.3f}"
                    f", probe_failed={metrics['probe_failed'].float().mean().item():.3f}"
                    f", probe_drop_hold={metrics['probe_drop_counter'].float().mean().item():.1f}"
                    f"/{self.cfg.grasp_probe_drop_hold_steps}"
                )
            finger_parts = []
            for finger_id, finger_name in enumerate(self.fingertip_log_names):
                rel_pos = metrics["fingertip_rel_pos"][:, finger_id].mean(dim=0)
                finger_parts.append(
                    f"{finger_name}:dist={metrics['fingertip_dist'][:, finger_id].mean().item():.4f},"
                    f"tip_force={metrics['fingertip_contact_force'][:, finger_id].mean().item():.4f},"
                    f"link_force={metrics['finger_contact_force'][:, finger_id].mean().item():.4f},"
                    f"rel=({rel_pos[0].item():+.4f},{rel_pos[1].item():+.4f},{rel_pos[2].item():+.4f})"
                )
            suffix += ", fingers " + "; ".join(finger_parts)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] current cache size: {total}, finished: {finished}{suffix}")

    def _is_cache_complete(self) -> bool:
        return all(saved.shape[0] >= self.grasp_cache_target_per_scale for saved in self.saved_grasping_states)

    def _write_cache_and_exit(self) -> None:
        save_data = torch.cat(
            [saved[: self.grasp_cache_target_per_scale] for saved in self.saved_grasping_states],
            dim=0,
        )
        output_path = self.cfg.grasp_output_path
        output_path = allegro_grasp_cache_path(output_path, self.cfg.scale_range)
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, save_data.cpu().numpy().astype(np.float32, copy=False))
        print(f"done! saved {save_data.shape[0]} Allegro grasp states to {output_path}")
        raise SystemExit(0)

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def _grasp_contact_forces(self) -> torch.Tensor:
        return wp.to_torch(self.finger_contact_sensor.data.net_forces_w_history)[:, :, self.grasp_contact_sensor_ids]

    def _validate_static_sampling_cfg(self) -> None:
        expected_values = {
            "torque_control": False,
            "randomize_pd_gains": False,
            "randomize_friction": False,
            "randomize_com": False,
            "force_scale": 0.0,
            "binary_contact": False,
            "enable_contact_pos": False,
        }
        for name, expected in expected_values.items():
            actual = getattr(self.cfg, name)
            if actual != expected:
                raise ValueError(
                    f"Allegro grasp cache requires cfg.{name}={expected!r} for static zero-action sampling; "
                    f"got {actual!r}."
                )

    def _log_grasp_physics_cfg(self) -> None:
        rigid_props = getattr(self.cfg.object_cfg.spawn, "rigid_props", None)
        kinematic_enabled = getattr(rigid_props, "kinematic_enabled", None)
        disable_gravity = getattr(rigid_props, "disable_gravity", None)
        print(
            "[INFO] Allegro grasp object physics: "
            f"kinematic_enabled={kinematic_enabled}, disable_gravity={disable_gravity}, "
            f"sim.gravity={getattr(self.cfg.sim, 'gravity', None)}, "
            f"live_gravity={tuple(self.physics_sim_view.get_gravity())}, "
            f"gravity_curriculum={self.cfg.gravity_curriculum}, "
            f"grasp_gravity_switch_interval={self.cfg.grasp_gravity_switch_interval}. "
            "Failed grasp episodes reset immediately through _get_dones()."
        )


def quat_angle_diff(quat: torch.Tensor, ref_quat: torch.Tensor) -> torch.Tensor:
    delta = quat_mul(quat, quat_conjugate(ref_quat))
    delta = delta / torch.clamp(torch.linalg.norm(delta, dim=-1, keepdim=True), min=1.0e-6)
    return torch.linalg.norm(axis_angle_from_quat(delta), dim=-1)
