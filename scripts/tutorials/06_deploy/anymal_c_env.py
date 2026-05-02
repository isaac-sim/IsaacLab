# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: I001

from __future__ import annotations

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .anymal_c_env_cfg import AnymalCFlatEnvCfg, AnymalCRoughEnvCfg
from leapp import annotate  # isort: skip


class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        self._base_id, _ = self._contact_sensor.find_sensors("base")
        self._feet_ids, _ = self._contact_sensor.find_sensors(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_sensors(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch
        # start LEAPP annotations for outputs
        annotate.update_state(self.spec.id, {"previous_actions": actions})
        annotate.output_tensors(self.spec.id, {"processed_actions": self._processed_actions}, export_with="onnx-dynamo")
        # end LEAPP annotations for outputs

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w.torch[:, 2].unsqueeze(1)
                - self._height_scanner.data.ray_hits_w.torch[..., 2]
                - 0.5
            ).clip(-1.0, 1.0)
        # start LEAPP annotations for inputs
        # NOTE: height data is not used by the flat policy. not needed for this example
        root_lin_vel_b = annotate.input_tensors(self.spec.id, {"root_lin_vel_b": self._robot.data.root_lin_vel_b.torch})
        root_ang_vel_b = annotate.input_tensors(self.spec.id, {"root_ang_vel_b": self._robot.data.root_ang_vel_b.torch})
        projected_gravity_b = annotate.input_tensors(
            self.spec.id, {"projected_gravity_b": self._robot.data.projected_gravity_b.torch}
        )
        commands = annotate.input_tensors(self.spec.id, {"commands": self._commands})
        joint_pos = annotate.input_tensors(self.spec.id, {"joint_pos": self._robot.data.joint_pos.torch})
        default_joint_pos = annotate.input_tensors(
            self.spec.id, {"default_joint_pos": self._robot.data.default_joint_pos.torch}
        )
        joint_vel = annotate.input_tensors(self.spec.id, {"joint_vel": self._robot.data.joint_vel.torch})
        previous_actions = annotate.state_tensors(self.spec.id, {"previous_actions": self._actions})
        # end LEAPP annotations for inputs

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    root_lin_vel_b,
                    root_ang_vel_b,
                    projected_gravity_b,
                    commands,
                    joint_pos - default_joint_pos,
                    joint_vel,
                    height_data,
                    previous_actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b.torch[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b.torch[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b.torch[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b.torch[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque.torch), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc.torch), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt).torch[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time.torch[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1
        )
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        is_contact = (
            torch.max(torch.linalg.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b.torch[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        died = torch.any(
            torch.max(torch.linalg.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        assert env_ids is not None
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids]
        joint_vel = self._robot.data.default_joint_vel.torch[env_ids]
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]
        default_root_pose[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
