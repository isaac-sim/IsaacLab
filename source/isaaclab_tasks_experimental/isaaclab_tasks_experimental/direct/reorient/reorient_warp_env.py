# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first Direct environment for the cube reorientation tasks.

Experimental counterpart of the torch-first mainline implementation in
:mod:`isaaclab_tasks.core.reorient`: environment-owned state lives in Warp arrays
and rewards, observations, and resets are computed in Warp kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import JointWrenchSensor, JointWrenchSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from isaaclab_tasks.core.reorient.reorient_task_base import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.core.utils import EpisodeErrorRecorder, randomize_rotation, sample_joint_positions_within_limits

from isaaclab_tasks_experimental.direct.reorient.reorient_kernels import (
    ReorientRewardBuffers,
    ema_actuation_kernel,
    full_obs_kernel,
    out_of_reach_kernel,
    reduced_obs_kernel,
    reorient_progress_kernel,
    reorient_reward,
    reorient_success_kernel,
)

if TYPE_CHECKING:
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


class ReorientWarpEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- Robot introspection: joints, bodies, limits --
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
        # warp-owned copies of the static joint limits (split into per-bound planes)
        self.hand_dof_lower_limits = wp.clone(wp.from_torch(joint_pos_limits[..., 0].contiguous()))
        self.hand_dof_upper_limits = wp.clone(wp.from_torch(joint_pos_limits[..., 1].contiguous()))

        # -- Observation layout, derived from the introspection --
        num_joints, num_fingers = self.num_hand_dofs, self.num_fingertips
        num_actions = self.cfg.action_space
        full_dim = 2 * num_joints + 13 + 11 + 13 * num_fingers + num_actions
        reduced_dim = 3 * num_fingers + 7 + num_actions
        state_dim = full_dim + 6 * num_fingers

        # -- Warp buffers: actuation --
        # joint/body index arrays consumed by the indexed kernels
        self.actuated_dof_ids = wp.array(self.actuated_dof_indices, dtype=wp.int32, device=str(self.device))
        self.finger_ids = wp.array(self.finger_bodies, dtype=wp.int32, device=str(self.device))
        self.wrench_ids = wp.array(
            self.finger_wrench_bodies if len(self.finger_wrench_bodies) else self.finger_bodies,
            dtype=wp.int32,
            device=str(self.device),
        )
        # EMA-smoothed joint position targets: full joint set plus the actuated-subset write buffer
        self.prev_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=str(self.device))
        self.cur_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=str(self.device))
        self.compact_targets = wp.zeros(
            (self.num_envs, len(self.actuated_dof_indices)), dtype=wp.float32, device=str(self.device)
        )
        # the policy action is copied in-place each step so the alias below stays bound
        self.actions = wp.zeros((self.num_envs, num_actions), dtype=wp.float32, device=str(self.device))

        # -- Warp buffers: goal and success state --
        # goal pose: the in-hand anchor and the sampled goal orientation / marker position
        self.in_hand_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=str(self.device))
        self.goal_rot = wp.zeros(self.num_envs, dtype=wp.quatf, device=str(self.device))
        self.goal_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=str(self.device))
        # success tracking: per-episode counts, goal-resample flags, and logging state
        self.successes = wp.zeros(self.num_envs, dtype=wp.float32, device=str(self.device))
        self.reset_goal_buf = wp.zeros(self.num_envs, dtype=wp.bool, device=str(self.device))
        self.consecutive_successes = wp.zeros(1, dtype=wp.float32, device=str(self.device))
        self._last_episode_success = wp.zeros(self.num_envs, dtype=wp.bool, device=str(self.device))

        # -- Warp buffers: per-step evaluation outputs --
        # success evaluation, written once per step in :meth:`_get_dones`
        self.success_flags = wp.empty(self.num_envs, dtype=wp.bool, device=str(self.device))
        self.orientation_error_buf = wp.empty(self.num_envs, dtype=wp.float32, device=str(self.device))
        # termination flags
        self.out_of_reach_flags = wp.empty(self.num_envs, dtype=wp.bool, device=str(self.device))
        self.time_out_flags = wp.empty(self.num_envs, dtype=wp.bool, device=str(self.device))
        # reward and diagnostics machinery (Warp-native internally)
        self._reward_buffers = ReorientRewardBuffers(self.num_envs, self.device)
        self._orientation_error = EpisodeErrorRecorder(self.num_envs, self.device)

        # -- Warp buffers: observation outputs --
        self.policy_obs_buf = wp.empty(
            (self.num_envs, reduced_dim if self.cfg.obs_type == "openai" else full_dim),
            dtype=wp.float32,
            device=str(self.device),
        )
        self.state_obs_buf = wp.empty((self.num_envs, state_dim), dtype=wp.float32, device=str(self.device))
        # placeholder for kernel launches while the wrench sensor has no data
        self.dummy_wrench = wp.zeros((1, 1), dtype=wp.vec3f, device=str(self.device))

        # -- Warp buffers: reset randomization constants --
        # batched unit axes consumed (through their torch aliases) by the torch.jit
        # reset helper; a Warp reset path would replace them with wp.vec3f constants
        self.x_unit_vec = wp.zeros(self.num_envs, dtype=wp.vec3f, device=str(self.device))
        self.y_unit_vec = wp.zeros(self.num_envs, dtype=wp.vec3f, device=str(self.device))
        self.z_unit_vec = wp.zeros(self.num_envs, dtype=wp.vec3f, device=str(self.device))

        # -- Visualization and articulation write handles --
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        self._set_joint_pos_target = self.hand.set_joint_position_target_index
        self._write_obj_root_pose = self.object.write_root_pose_to_sim_index
        self._write_obj_root_vel = self.object.write_root_velocity_to_sim_index
        self._write_hand_joint_pos = self.hand.write_joint_position_to_sim_index
        self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim_index

        # -- Bind torch aliases to the warp-native buffers (zero-copy views) --
        # robot joint limits
        self.hand_dof_lower_limits_torch = wp.to_torch(self.hand_dof_lower_limits)
        self.hand_dof_upper_limits_torch = wp.to_torch(self.hand_dof_upper_limits)
        # actuation
        self.prev_targets_torch = wp.to_torch(self.prev_targets)
        self.cur_targets_torch = wp.to_torch(self.cur_targets)
        self.compact_targets_torch = wp.to_torch(self.compact_targets)
        self.actions_torch = wp.to_torch(self.actions)
        # goal and success state
        self.in_hand_pos_torch = wp.to_torch(self.in_hand_pos)
        self.goal_rot_torch = wp.to_torch(self.goal_rot)
        self.goal_pos_torch = wp.to_torch(self.goal_pos)
        self.successes_torch = wp.to_torch(self.successes)
        self.reset_goal_buf_torch = wp.to_torch(self.reset_goal_buf)
        self.consecutive_successes_torch = wp.to_torch(self.consecutive_successes)
        self._last_episode_success_torch = wp.to_torch(self._last_episode_success)
        # per-step evaluation and reward outputs
        self.out_of_reach_flags_torch = wp.to_torch(self.out_of_reach_flags)
        self.time_out_flags_torch = wp.to_torch(self.time_out_flags)
        self.reward_torch = wp.to_torch(self._reward_buffers.reward)
        # observation outputs
        self.policy_obs_buf_torch = wp.to_torch(self.policy_obs_buf)
        self.state_obs_buf_torch = wp.to_torch(self.state_obs_buf)
        # reset randomization constants
        self.x_unit_vec_torch = wp.to_torch(self.x_unit_vec)
        self.y_unit_vec_torch = wp.to_torch(self.y_unit_vec)
        self.z_unit_vec_torch = wp.to_torch(self.z_unit_vec)

        # -- Cached warp views of externally-owned torch tensors --
        self.episode_length_buf_warp = wp.from_torch(self.episode_length_buf)
        self.env_origins_warp = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # -- Initial buffer contents (written through the torch aliases) --
        # in-hand target = object default position + shared offset (mirrors ReorientCommand)
        self.in_hand_pos_torch[:] = self.object.data.default_root_pose.torch[:, 0:3]
        self.in_hand_pos_torch += torch.tensor(
            IN_HAND_POS_OFFSET, dtype=self.in_hand_pos_torch.dtype, device=self.in_hand_pos_torch.device
        )
        self.goal_rot_torch[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout
        self.goal_pos_torch[:, :] = torch.tensor(GOAL_MARKER_POSITION, device=self.device)
        self.x_unit_vec_torch[:] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.y_unit_vec_torch[:] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.z_unit_vec_torch[:] = torch.tensor([0.0, 0.0, 1.0], device=self.device)

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
        # copy in-place: keeps the persistent warp-native action buffer (and its
        # cached kernel view) bound to the same memory across steps
        self.actions_torch.copy_(actions)

    def _apply_action(self) -> None:
        wp.launch(
            ema_actuation_kernel,
            dim=(self.num_envs, self.actuated_dof_ids.shape[0]),
            inputs=[
                self.actions,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.actuated_dof_ids,
                self.cfg.act_moving_average,
                self.prev_targets,
                self.cur_targets,
            ],
            outputs=[self.compact_targets],
            device=self.compact_targets.device,
        )
        self._set_joint_pos_target(target=self.compact_targets_torch, joint_ids=self.actuated_dof_indices)

    def _get_observations(self) -> dict:
        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            raise ValueError(f"Unknown observation type: {self.cfg.obs_type}. Should be 'full' or 'openai'.")

        # RSL-RL holds the observation reference across the next env.step, so hand out
        # a per-step snapshot rather than the persistent kernel output buffer.
        # TODO: drop the per-step clone by double-buffering the observation outputs
        # (pre-allocated alternating Warp buffers with cached torch bindings).
        observations = {"policy": obs.clone()}
        if self.cfg.asymmetric_obs:
            observations["critic"] = self.compute_full_state().clone()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # success flags, distances, and the episode-minimum error tracking were
        # already computed this step by the kernel launches in :meth:`_get_dones`;
        # the goal-reset flags, success counts, and moving average update in place.
        # passing the time-out flags alone is exact here: the kernel ORs in the
        # fallen-object condition itself, and that is this env's only other reset
        reorient_reward(
            self.time_out_flags,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.object.data.root_pos_w.warp,
            self.env_origins_warp,
            self.object.data.root_quat_w.warp,
            self.in_hand_pos,
            self.goal_rot,
            self.actions,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            self._reward_buffers,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes_torch.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf_torch.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return self.reward_torch

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # reset when cube has fallen
        wp.launch(
            out_of_reach_kernel,
            dim=self.num_envs,
            inputs=[
                self.object.data.root_pos_w.warp,
                self.env_origins_warp,
                self.in_hand_pos,
                self.cfg.fall_dist,
            ],
            outputs=[self.out_of_reach_flags],
            device=self.out_of_reach_flags.device,
        )
        # single per-step success evaluation (rewards and metrics reuse the buffers),
        # followed by the progress bookkeeping (episode-counter reset in place,
        # time-out by length or consecutive-success cap, episode-minimum error tracking)
        wp.launch(
            reorient_success_kernel,
            dim=self.num_envs,
            inputs=[self.object.data.root_quat_w.warp, self.goal_rot, self.cfg.success_tolerance],
            outputs=[self.success_flags, self.orientation_error_buf],
            device=self.orientation_error_buf.device,
        )
        wp.launch(
            reorient_progress_kernel,
            dim=self.num_envs,
            inputs=[
                self.success_flags,
                self.orientation_error_buf,
                self.successes,
                float(self.cfg.max_consecutive_success),
                self.max_episode_length,
            ],
            outputs=[
                self.episode_length_buf_warp,
                self._orientation_error.minimum_error,
                self.time_out_flags,
                self._orientation_error._has_sample,
            ],
            device=self.orientation_error_buf.device,
        )
        return self.out_of_reach_flags_torch, self.time_out_flags_torch

    def _reset_idx(self, env_ids: Sequence[int]):
        # Episode counts as successful when goals reached >= cfg.success_count_threshold.
        self._last_episode_success_torch[env_ids] = self.successes_torch[env_ids] >= self.cfg.success_count_threshold
        # 0-dim device tensor: avoids a host sync here; consumers read it at logging cadence
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._last_episode_success_torch[env_ids].float().mean()
        )
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
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_vec_torch[env_ids], self.y_unit_vec_torch[env_ids]
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

        self.prev_targets_torch[env_ids] = dof_pos
        self.cur_targets_torch[env_ids] = dof_pos

        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=dof_vel, env_ids=env_ids)

        self.successes_torch[env_ids] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_vec_torch[env_ids], self.y_unit_vec_torch[env_ids]
        )

        # update goal pose and markers
        self.goal_rot_torch[env_ids] = new_rot
        goal_pos = self.goal_pos_torch + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot_torch)

        self.reset_goal_buf_torch[env_ids] = False

    def _compute_intermediate_values(self):
        """Refresh the torch-side state snapshots consumed by the camera environment's observations."""
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
        wp.launch(
            reduced_obs_kernel,
            dim=self.num_envs,
            inputs=[
                self.hand.data.body_pos_w.warp,
                self.env_origins_warp,
                self.finger_ids,
                self.object.data.root_pos_w.warp,
                self.object.data.root_quat_w.warp,
                self.goal_rot,
                self.actions,
            ],
            outputs=[self.policy_obs_buf],
            device=self.policy_obs_buf.device,
        )
        return self.policy_obs_buf_torch

    def compute_full_observations(self):
        force, torque = self.dummy_wrench, self.dummy_wrench
        wp.launch(
            full_obs_kernel,
            dim=(self.num_envs, self.policy_obs_buf.shape[1]),
            inputs=[
                self.hand.data.joint_pos.warp,
                self.hand.data.joint_vel.warp,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.object.data.root_pos_w.warp,
                self.env_origins_warp,
                self.object.data.root_quat_w.warp,
                self.object.data.root_lin_vel_w.warp,
                self.object.data.root_ang_vel_w.warp,
                self.in_hand_pos,
                self.goal_rot,
                self.hand.data.body_pos_w.warp,
                self.hand.data.body_quat_w.warp,
                self.hand.data.body_vel_w.warp,
                self.finger_ids,
                force,
                torque,
                self.wrench_ids,
                self.actions,
                self.cfg.vel_obs_scale,
                self.cfg.force_torque_obs_scale,
                0,
            ],
            outputs=[self.policy_obs_buf],
            device=self.policy_obs_buf.device,
        )
        return self.policy_obs_buf_torch

    def compute_full_state(self):
        force, torque, with_forces = self.dummy_wrench, self.dummy_wrench, -1
        if self._joint_wrench_sensor is not None:
            force_data = self._joint_wrench_sensor.data.force
            torque_data = self._joint_wrench_sensor.data.torque
            if force_data is not None and torque_data is not None:
                force, torque, with_forces = force_data.warp, torque_data.warp, 1
        wp.launch(
            full_obs_kernel,
            dim=(self.num_envs, self.state_obs_buf.shape[1]),
            inputs=[
                self.hand.data.joint_pos.warp,
                self.hand.data.joint_vel.warp,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.object.data.root_pos_w.warp,
                self.env_origins_warp,
                self.object.data.root_quat_w.warp,
                self.object.data.root_lin_vel_w.warp,
                self.object.data.root_ang_vel_w.warp,
                self.in_hand_pos,
                self.goal_rot,
                self.hand.data.body_pos_w.warp,
                self.hand.data.body_quat_w.warp,
                self.hand.data.body_vel_w.warp,
                self.finger_ids,
                force,
                torque,
                self.wrench_ids,
                self.actions,
                self.cfg.vel_obs_scale,
                self.cfg.force_torque_obs_scale,
                with_forces,
            ],
            outputs=[self.state_obs_buf],
            device=self.state_obs_buf.device,
        )
        return self.state_obs_buf_torch
