# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import JointWrenchSensor, JointWrenchSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform, saturate, scale_transform, unscale_transform

from isaaclab_tasks.core.reorient.mdp.rewards import evaluate_reorient_success, reorient_reward
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.core.utils import EpisodeErrorRecorder, randomize_rotation, sample_joint_positions_within_limits

if TYPE_CHECKING:
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg


class ReorientDirectEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- robot introspection: joints, bodies, limits --
        self.num_hand_dofs = self.hand.num_joints
        self.actuated_dof_indices, _ = self.hand.find_joints(cfg.actuated_joint_names)
        if len(self.actuated_dof_indices) != len(cfg.actuated_joint_names):
            raise ValueError(
                f"Expected {len(cfg.actuated_joint_names)} actuated joints, found {len(self.actuated_dof_indices)}."
            )
        self.finger_bodies, fingertip_body_names = self.hand.find_bodies(self.cfg.fingertip_body_names)
        if len(self.finger_bodies) != len(self.cfg.fingertip_body_names):
            raise ValueError(
                f"Expected {len(self.cfg.fingertip_body_names)} fingertip bodies, found {len(self.finger_bodies)}."
            )
        self.num_fingertips = len(self.finger_bodies)
        self.finger_wrench_bodies = []
        if getattr(self, "_joint_wrench_sensor", None) is not None:
            for body_name in fingertip_body_names:
                self.finger_wrench_bodies.append(self._joint_wrench_sensor.body_names.index(body_name))
            self.finger_wrench_bodies.sort()
        joint_pos_limits = self.hand.data.joint_limits.torch.to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # -- actuation targets (EMA-smoothed joint position targets) --
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # -- goal and success state --
        # in-hand target = object default position + shared offset (mirrors ReorientCommand)
        self.in_hand_pos = self.object.data.default_root_pose.torch[:, 0:3].clone()
        self.in_hand_pos += torch.tensor(IN_HAND_POS_OFFSET, dtype=torch.float, device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor(GOAL_MARKER_POSITION, device=self.device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self._last_episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- per-step evaluation state and diagnostics --
        # written once per step in :meth:`_get_dones`; the reward and metrics reuse them
        self._success_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._orientation_error_buf = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._orientation_error = EpisodeErrorRecorder(self.num_envs, self.device)

        # -- reset randomization constants --
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # -- visualization and articulation write handles --
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        self._set_joint_pos_target = self.hand.set_joint_position_target_index
        self._write_obj_root_pose = self.object.write_root_pose_to_sim_index
        self._write_obj_root_vel = self.object.write_root_velocity_to_sim_index
        self._write_hand_joint_pos = self.hand.write_joint_position_to_sim_index
        self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim_index

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object: Articulation | RigidObject = self.cfg.object_cfg.class_type(self.cfg.object_cfg)
        self._joint_wrench_sensor = None
        if self.cfg.asymmetric_obs:
            self._joint_wrench_sensor = self._create_joint_wrench_sensor()
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        if self._joint_wrench_sensor is not None:
            self.scene.sensors["joint_wrench"] = self._joint_wrench_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_joint_wrench_sensor(self) -> JointWrenchSensor:
        """Create the joint-wrench sensor used for fingertip force/torque observations."""
        return JointWrenchSensor(JointWrenchSensorCfg(prim_path=self.cfg.robot_cfg.prim_path))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = unscale_transform(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self._set_joint_pos_target(
            target=self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self._update_fingertip_force_sensors()

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            raise ValueError(f"Unknown observation type: {self.cfg.obs_type}. Should be 'full' or 'openai'.")

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations["critic"] = self.compute_full_state()
        return observations

    def _update_fingertip_force_sensors(self) -> None:
        """Update fingertip force/torque observations from the joint-wrench sensor."""
        if getattr(self, "_joint_wrench_sensor", None) is None:
            self.fingertip_force_sensors = torch.zeros(
                self.num_envs, len(self.finger_bodies), 6, dtype=torch.float32, device=self.device
            )
            return

        sensor_data = self._joint_wrench_sensor.data
        force_data = sensor_data.force
        torque_data = sensor_data.torque
        if force_data is None or torque_data is None:
            self.fingertip_force_sensors = torch.zeros(
                self.num_envs, len(self.finger_bodies), 6, dtype=torch.float32, device=self.device
            )
            return

        force = force_data.torch[:, self.finger_wrench_bodies]
        torque = torque_data.torch[:, self.finger_wrench_bodies]
        self.fingertip_force_sensors = torch.cat((force, torque), dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        # the success flags and orientation errors were computed this step by
        # :meth:`_get_dones`; the recorder and the reward reuse them
        self._orientation_error.update(self._orientation_error_buf)
        total_reward, goal_resets, successes, consecutive_successes = reorient_reward(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.object_pos,
            self.in_hand_pos,
            self._success_flags,
            self._orientation_error_buf,
            self.actions,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.cfg.action_penalty_scale,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )
        self.reset_goal_buf.copy_(goal_resets)
        self.successes[:] = successes
        self.consecutive_successes[:] = consecutive_successes

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.linalg.norm(self.object_pos - self.in_hand_pos, ord=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        # single per-step success evaluation; the reward and metrics reuse these buffers
        self._success_flags, self._orientation_error_buf = evaluate_reorient_success(
            self.object_rot, self.goal_rot, self.cfg.success_tolerance
        )

        if self.cfg.max_consecutive_success > 0:
            # reset progress (episode length buf) on goal environments
            self.episode_length_buf = torch.where(
                self._success_flags,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        # Episode counts as successful when goals reached >= cfg.success_count_threshold.
        self._last_episode_success[env_ids] = self.successes[env_ids] >= self.cfg.success_count_threshold
        # 0-dim device tensor: avoids a host sync here; consumers read it at logging cadence
        self.extras.setdefault("log", {})["Metrics/success_rate"] = self._last_episode_success[env_ids].float().mean()
        for statistic, value in self._orientation_error.reset(env_ids).items():
            self.extras["log"][f"Diagnostics/episode_min_orientation_error_{statistic}"] = value

        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_pose = self.object.data.default_root_pose.torch.clone()[env_ids]
        object_default_vel = self.object.data.default_root_vel.torch.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_pose[:, 0:3] = (
            object_default_pose[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_pose[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_vel[:] = 0.0
        self._write_obj_root_pose(root_pose=object_default_pose, env_ids=env_ids)
        self._write_obj_root_vel(root_velocity=object_default_vel, env_ids=env_ids)

        # reset hand
        default_dof_pos = self.hand.data.default_joint_pos.torch[env_ids]
        dof_limits = self.hand.data.joint_limits.torch[env_ids]
        dof_pos = sample_joint_positions_within_limits(default_dof_pos, dof_limits, self.cfg.reset_dof_pos_noise)

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel.torch[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        """Refresh the torch-side state snapshots consumed by the observation and reward paths."""
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w.torch[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch

        # data for object
        self.object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w.torch
        self.object_velocities = self.object.data.root_vel_w.torch
        self.object_linvel = self.object.data.root_lin_vel_w.torch
        self.object_angvel = self.object.data.root_ang_vel_w.torch

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                scale_transform(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                scale_transform(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states
