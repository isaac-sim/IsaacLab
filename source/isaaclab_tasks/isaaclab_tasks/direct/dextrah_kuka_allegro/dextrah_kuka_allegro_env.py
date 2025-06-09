# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform, normalize

from .dextrah_kuka_allegro_env_cfg import DextrahKukaAllegroEnvCfg
from .dextrah_adr import DextrahADR


class DextrahKukaAllegroEnv(DirectRLEnv):
    cfg: DextrahKukaAllegroEnvCfg

    def __init__(self, cfg: DextrahKukaAllegroEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.cfg.robot_scene_cfg.resolve(self.scene)
        self.object_goal = torch.tensor(self.cfg.object_goal, device=self.device).repeat((self.num_envs, 1))
        self.object_goal += self.scene.env_origins
        self.curled_q = torch.tensor(self.cfg.curled_q, device=self.device).repeat(self.num_envs, 1).contiguous()
        self.joint_pos_action = self.cfg.joint_pos_action_cfg.class_type(self.cfg.joint_pos_action_cfg, self)
        self.joint_vel_action = self.cfg.joint_vel_action_cfg.class_type(self.cfg.joint_vel_action_cfg, self)

        # Set up ADR
        self.dextrah_adr = DextrahADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
        self.step_since_last_dr_change = 0
        self.dextrah_adr.set_num_increments(self.cfg.starting_adr_increments)
        self.local_adr_increment = torch.tensor(self.cfg.starting_adr_increments, device=self.device, dtype=torch.int64)
        self.global_min_adr_increment = self.local_adr_increment.clone()

        # Track success statistics
        self.time_in_success_region = torch.zeros(self.num_envs, device=self.device)

        # wrench
        self.object_applied_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_applied_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Object noise
        self.object_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_vel_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_vel_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        center = self.cfg.objects_cfg.init_state.pos
        self.oob_limits = torch.tensor([
            [center[0] - self.cfg.obj_spawn_width[0] / 2., center[1] - self.cfg.obj_spawn_width[1] / 2., 0.2],
            [center[0] + self.cfg.obj_spawn_width[0] / 2., center[1] + self.cfg.obj_spawn_width[1] / 2., float('inf')]
        ], device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["hand_to_object", "object_to_goal", "finger_curl_reg", "lift_reward"]
        }

    def _setup_scene(self):
        # add robot, objects
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.objects_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        num_unique_objects = len(self.object.cfg.spawn.assets_cfg)
        num_teacher_observations = 148 + num_unique_objects
        self.cfg.state_space = 180 + num_unique_objects
        self.cfg.observation_space = num_teacher_observations

        self.multi_object_idx = torch.remainder(torch.arange(self.num_envs), num_unique_objects).to(self.device)
        self.multi_object_idx_onehot = F.one_hot(self.multi_object_idx, num_classes=num_unique_objects).float()
        self.object_scale = torch.ones((self.num_envs, 1), device=self.device)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Find the current global minimum adr increment
        local_adr_increment = self.local_adr_increment.clone()
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.all_reduce(local_adr_increment, op=dist.ReduceOp.MIN)
        self.global_min_adr_increment = local_adr_increment

        self._actions = actions.clone()
        self.joint_pos_action.process_actions(actions[:, :23])
        self.joint_vel_action.process_actions(actions[:, 23:])
        self.apply_object_wrench()

    def _apply_action(self) -> None:
        self.joint_pos_action.apply_actions()
        self.joint_vel_action.apply_actions()

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self.cfg.robot_scene_cfg.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.cfg.robot_scene_cfg.joint_ids]
        hand_pos = self.robot.data.body_pos_w[:, self.cfg.robot_scene_cfg.body_ids].view(self.num_envs, -1)
        hand_pos -= self.scene.env_origins.repeat((1, len(self.cfg.robot_scene_cfg.body_ids)))
        hand_vel = self.robot.data.body_vel_w[:, self.cfg.robot_scene_cfg.body_ids].view(self.num_envs, -1)
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        object_rot = self.object.data.root_quat_w
        hand_forces = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.cfg.robot_scene_cfg.body_ids]
        measured_joint_torques = self.robot.root_physx_view.get_dof_projected_joint_forces()

        joint_pos_noisy = joint_pos + self.joint_pos_noise_width * rand_like(joint_pos) + self.joint_pos_bias
        joint_vel_noisy = joint_vel + self.joint_vel_noise_width * rand_like(joint_pos) + self.joint_vel_bias
        joint_vel_noisy *= self.dextrah_adr.get_custom_param_value("observation_annealing", "coefficient")

        hand_pos_noisy = self.robot.data.body_pos_w[:, self.cfg.robot_scene_cfg.body_ids].view(self.num_envs, -1)
        hand_pos_noisy -= self.scene.env_origins.repeat((1, len(self.cfg.robot_scene_cfg.body_ids)))
        hand_vel_noisy = self.robot.data.body_vel_w[:, self.cfg.robot_scene_cfg.body_ids].view(self.num_envs, -1)
        hand_vel_noisy *= self.dextrah_adr.get_custom_param_value("observation_annealing", "coefficient")

        object_pos_noisy = object_pos + self.object_pos_noise_width * rand_like(object_pos) + self.object_pos_bias
        object_rot_noisy = object_rot + self.object_rot_noise_width * rand_like(object_rot) + self.object_rot_bias

        teacher_policy_obs = torch.cat(
            (
                joint_pos_noisy,  # 0:23
                joint_vel_noisy,  # 23:46
                hand_pos_noisy,  # 46:61
                hand_vel_noisy,  # 61:91
                object_pos_noisy,  # 91:94
                object_rot_noisy,  # 94:98
                self.object_goal - self.scene.env_origins,  # 98:101
                self.multi_object_idx_onehot,  # 101:253
                self.object_scale,  # 253:254
                self._actions,  # 254:300
            ), dim=-1,
        )

        critic_obs = torch.cat(
            (
                joint_pos,  # 0:23
                joint_vel,  # 23:46
                hand_pos,  # 46:61
                hand_vel,  # 61:76
                hand_forces.view(self.num_envs, -1)[:, :3],
                measured_joint_torques,
                object_pos,
                object_rot,
                self.object.data.root_vel_w,
                self.object_goal - self.scene.env_origins,
                self.multi_object_idx_onehot,
                self.object_scale,
                self._actions,
            ), dim=-1,
        )

        observations = {"policy": teacher_policy_obs, "critic": critic_obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        hand_pos = self.robot.data.body_pos_w[:, self.cfg.robot_scene_cfg.body_ids]
        object_pos = self.object.data.root_pos_w
        joint_pos = self.robot.data.joint_pos[:, self.cfg.robot_scene_cfg.joint_ids]

        object_to_goal_pos_error = torch.norm(object_pos - self.object_goal, dim=-1)
        object_vertical_error = torch.abs(self.object_goal[:, 2] - object_pos[:, 2])
        hand_to_object_pos_error = torch.norm(hand_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
        in_success_region = torch.norm(object_pos - self.object_goal, dim=-1) < self.cfg.object_goal_tol

        object_to_goal_std = self.dextrah_adr.get_custom_param_value("reward_weights", "object_to_goal_sharpness")
        finger_curl_reg_weight = self.dextrah_adr.get_custom_param_value("reward_weights", "finger_curl_reg")
        lift_weight = self.dextrah_adr.get_custom_param_value("reward_weights", "lift_weight")

        hand_to_object_rew = torch.exp(-self.cfg.hand_to_object_sharpness * hand_to_object_pos_error)
        object_to_goal_rew = torch.exp(-object_to_goal_std * object_to_goal_pos_error)
        finger_curl_reg = torch.sum(torch.square(joint_pos[:, 7:] - self.curled_q), dim=-1)
        lift_rew = torch.exp(-self.cfg.lift_sharpness * object_vertical_error)

        # Add reward signals to tensorboard
        self.extras["num_adr_increases"] = self.dextrah_adr.num_increments()
        self.extras["in_success_region"] = in_success_region.float().mean()
        self.extras["true_objective"] = self.true_objective
        self.extras["true_objective_mean"] = self.true_objective.float().mean()
        self.extras["true_objective_min"] = self.true_objective.float().min()
        self.extras["true_objective_max"] = self.true_objective.float().max()

        rewards = {
            "hand_to_object" : self.cfg.hand_to_object_weight * hand_to_object_rew * self.step_dt,
            "object_to_goal" : self.cfg.object_to_goal_weight * object_to_goal_rew * self.step_dt,
            "finger_curl_reg" : finger_curl_reg_weight * finger_curl_reg * self.step_dt,
            "lift_reward" : lift_weight * lift_rew * self.step_dt
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # This should be in starte
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # bookkeeping for ADR
        in_success_region = torch.norm(object_pos - self.object_goal, dim=-1) < self.cfg.object_goal_tol
        self.time_in_success_region = torch.where(in_success_region, self.time_in_success_region + self.cfg.sim.dt*self.cfg.decimation, 0.)
        if self.dextrah_adr.num_increments() <= 49:
            self.true_objective = self.dextrah_adr.num_increments() + 0.0 * in_success_region.float()
        else:
            self.true_objective = self.dextrah_adr.num_increments() + 50.0 * in_success_region.float().mean() * torch.ones_like(in_success_region).float()

        outside_bounds = ((object_pos < self.oob_limits[0]) | (object_pos > self.oob_limits[1])).any(dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return outside_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Update DR ranges this needs to happen before reset
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        in_success_region = torch.norm(object_pos - self.object_goal, dim=-1) < self.cfg.object_goal_tol
        if self.cfg.enable_adr:
            time_elapsed = self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change
            metric_met = in_success_region.float().mean() > self.cfg.success_for_adr
            all_gpu_synced = self.local_adr_increment == self.global_min_adr_increment
            if time_elapsed and metric_met and all_gpu_synced:
                self.step_since_last_dr_change = 0
                self.dextrah_adr.increase_ranges(increase_counter=True)
                self.event_manager.reset(env_ids=self.robot._ALL_INDICES)
                self.event_manager.apply(env_ids=self.robot._ALL_INDICES, mode="reset", global_env_step_count=0)
                self.local_adr_increment = torch.tensor(self.dextrah_adr.num_increments(), device=self.device, dtype=torch.int64)
            else:
                self.step_since_last_dr_change += 1

        super()._reset_idx(env_ids)
        num_ids = env_ids.shape[0]
        # Reset object state
        object_start_state = self.object.data.default_root_state[env_ids].clone()
        # apply translation
        xy_width = [self.dextrah_adr.get_custom_param_value("object_spawn", "x_width_spawn"), self.dextrah_adr.get_custom_param_value("object_spawn", "y_width_spawn")]
        object_start_state[:, :2] += (rand_like(object_start_state[:, :2]) / 2) * torch.tensor(xy_width, device=self.device)
        object_start_state[:, :3] += self.scene.env_origins[env_ids]
        # apply orientation
        rotation = self.dextrah_adr.get_custom_param_value("object_spawn", "rotation")
        rot_noise = sample_uniform(-rotation, rotation, (num_ids, 2), device=self.device)  # noise for X and Y rotation
        object_start_state[:, 3:7] = quat_mul(
            quat_from_angle_axis(rot_noise[:, 0] * 3.14, torch.tensor([1., 0., 0.], device=self.device).repeat((num_ids, 1))),
            quat_from_angle_axis(rot_noise[:, 1] * 3.14, torch.tensor([0., 1., 0.], device=self.device).repeat((num_ids, 1)))
        )

        self.object.write_root_state_to_sim(object_start_state, env_ids)

        # Spawning robot
        joint_pos_noise = self.dextrah_adr.get_custom_param_value("robot_spawn" ,"joint_pos_noise")
        joint_vel_noise = self.dextrah_adr.get_custom_param_value("robot_spawn" ,"joint_vel_noise")
        default_joint_pos = self.robot.data.default_joint_pos[env_ids[:, None], self.cfg.robot_scene_cfg.joint_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids[:, None], self.cfg.robot_scene_cfg.joint_ids]

        # Calculate joint positions
        dof_pos = (joint_pos_noise * rand_like(default_joint_pos) + default_joint_pos).clamp(
            min=self.robot.data.joint_pos_limits[:, self.cfg.robot_scene_cfg.joint_ids, 0][0],
            max=self.robot.data.joint_pos_limits[:, self.cfg.robot_scene_cfg.joint_ids, 1][0]
        )

        dof_vel = joint_vel_noise * rand_like(default_joint_vel) + default_joint_vel

        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids, joint_ids=self.cfg.robot_scene_cfg.joint_ids)
        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids, joint_ids=self.cfg.robot_scene_cfg.joint_ids)
        self.robot.set_joint_velocity_target(dof_vel, env_ids=env_ids, joint_ids=self.cfg.robot_scene_cfg.joint_ids)

        # Reset success signals
        self.time_in_success_region[env_ids] = 0.

        # NOTE: CPU Operations, so we only query infrequently
        self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device)
        self.robot_dof_stiffness = self.robot.root_physx_view.get_dof_stiffnesses().to(device=self.device)
        self.robot_dof_damping = self.robot.root_physx_view.get_dof_dampings().to(device=self.device)

        # OBJECT NOISE---------------------------------------------------------------------------
        rand = lambda : torch.rand((num_ids, 1), device=self.device)
        object_pos_bias_width = self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_bias") * rand()
        object_rot_bias_width = self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_bias") * rand()
        self.object_pos_bias[env_ids, :] = object_pos_bias_width * (rand() - 0.5)
        self.object_rot_bias[env_ids, :] = object_rot_bias_width * (rand() - 0.5)

        # TODO: alternative (need to check with ankur)
        # self.object_pos_bias[env_ids, :] = self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_bias") * (rand() - 0.5)
        # self.object_rot_bias[env_ids, :] = self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_bias") * (rand() - 0.5)

        # Sample width of per-step noise
        self.object_pos_noise_width[env_ids, :] = self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_noise") * rand()
        self.object_rot_noise_width[env_ids, :] = self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_noise") * rand()

        # ROBOT NOISE---------------------------------------------------------------------------
        # Sample widths of uniform distribution controlling robot state bias
        joint_pos_bias_width = self.dextrah_adr.get_custom_param_value("robot_state_noise", "joint_pos_bias") * rand()
        joint_vel_bias_width = self.dextrah_adr.get_custom_param_value("robot_state_noise", "joint_vel_bias") * rand()
        self.joint_pos_bias[env_ids, :] = joint_pos_bias_width * (rand() - 0.5)
        self.joint_vel_bias[env_ids, :] = joint_vel_bias_width * (rand() - 0.5)

        # TODO: alternative (need to check with ankur)
        # self.joint_pos_bias[env_ids, :] = self.dextrah_adr.get_custom_param_value("robot_state_noise", "joint_pos_bias") * (rand() - 0.5)
        # self.joint_vel_bias[env_ids, :] = self.dextrah_adr.get_custom_param_value("robot_state_noise", "joint_vel_bias") * (rand() - 0.5)

        # Sample width of per-step noise
        self.joint_pos_noise_width[env_ids, :] = self.dextrah_adr.get_custom_param_value("robot_state_noise", "joint_pos_noise") * rand()
        self.joint_vel_noise_width[env_ids, :] = self.dextrah_adr.get_custom_param_value("robot_state_noise", "joint_vel_noise") * rand()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def apply_object_wrench(self):
        hand_pos = self.robot.data.body_pos_w[:, self.cfg.robot_scene_cfg.body_ids]
        object_pos = self.object.data.root_pos_w - self.scene.env_origins

        # Update whether to apply wrench based on whether object is at goal
        hand_to_object_pos_error = torch.norm(hand_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
        num_bodies = self.object.num_bodies
        # Generates the random wrench
        max_linear_accel = self.dextrah_adr.get_custom_param_value("object_wrench", "max_linear_accel")
        acc_scalar = torch.rand(self.num_envs, 1, device=self.device)
        max_force = (max_linear_accel * self.object_mass).unsqueeze(2)
        max_torque = (self.object_mass * max_linear_accel * self.cfg.torsional_radius).unsqueeze(2)
        forces = max_force * acc_scalar * normalize(torch.randn(self.num_envs, num_bodies, 3, device=self.device))
        torques = max_torque * acc_scalar * normalize(torch.randn(self.num_envs, num_bodies, 3, device=self.device))

        wrench_triggered = (self.episode_length_buf.view(-1, 1, 1) % self.cfg.wrench_trigger_every) == 0
        apply_wrench_mask = (hand_to_object_pos_error <= self.cfg.hand_to_object_dist_threshold)[:, None, None]

        self.object_applied_force = torch.where(apply_wrench_mask, self.object_applied_force, torch.zeros_like(self.object_applied_force))
        self.object_applied_torque = torch.where(apply_wrench_mask, self.object_applied_torque, torch.zeros_like(self.object_applied_torque))

        self.object_applied_force = torch.where(wrench_triggered, forces, self.object_applied_force)
        self.object_applied_torque = torch.where(wrench_triggered, torques, self.object_applied_torque)

        # Set the wrench to the buffers
        self.object.set_external_force_and_torque(forces=self.object_applied_force, torques=self.object_applied_torque)
        self.object.write_data_to_sim()  # TODO: check this needed? I feel like this can be buggy


def rand_like(tensor_in):
    return 2. * (torch.rand_like(tensor_in) - 0.5)
