# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from .drone_env_cfg import DroneEnvCfg
from .thruster_cfg import ThrusterCfg
from .controller_cfg import LeeControllerCfg

class DroneEnv(DirectRLEnv):
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.external_wrench = torch.zeros((self.num_envs, 9, 6), device=self.device)
        # Logging
        # self._episode_sums = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in ["success_by_distance", "behavior shaping", "early termination"]}
        self._reward_buf = torch.zeros((self.num_envs,), device=self.device)
        self._terminated = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._truncated = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        
        # drag
        self.body_vel_linear_damping_coefficient = torch.tensor(self.cfg.body_vel_linear_damping_coefficient).to(self.device)
        self.body_vel_quadratic_damping_coefficient = torch.tensor(self.cfg.body_vel_quadratic_damping_coefficient).to(self.device)
        self.angvel_linear_damping_coefficient = torch.tensor(self.cfg.angvel_linear_damping_coefficient).to(self.device)
        self.angvel_quadratic_damping_coefficient = torch.tensor(self.cfg.angvel_quadratic_damping_coefficient).to(self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        
    def _pre_physics_step(self, actions: torch.Tensor):
        if self._sim_step_counter == 0:
            # Controller
            controller_cfg = LeeControllerCfg()
            self.controller = controller_cfg.class_type(controller_cfg, self)
            
            # Thruster
            self.motor_directions = torch.tensor(self.cfg.motor_directions, device=self.device)
            self.inv_force_torque_allocation_matrix = torch.linalg.pinv(torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)).expand(self.num_envs, -1, -1)
            thruster_cfg = ThrusterCfg(dt=self.cfg.sim.dt)
            self.thruster = thruster_cfg.class_type(self.num_envs, cfg=thruster_cfg, device=self.device)
            self.thrust_to_torque_ratio = self.cfg.thrust_to_torque_ratio
        
        self._actions = actions.clone()
        self._processed_actions = actions.clamp(-10, 10)
        controller_output = self.controller(self._processed_actions)
        
        # call actuator model to get forces and torque
        ref_motor_thrusts = torch.bmm(self.inv_force_torque_allocation_matrix, controller_output.unsqueeze(-1)).squeeze(-1)
        motor_thrusts = self.thruster.update_motor_thrusts(ref_motor_thrusts)
        zero_thrust = torch.zeros_like(motor_thrusts)
        motor_forces = torch.stack((zero_thrust, zero_thrust, motor_thrusts), dim=2)
        motor_torques = self.thrust_to_torque_ratio * motor_forces * (-self.motor_directions[None, :, None])
        
        self.external_wrench[:, self.cfg.application_mask, :3] = motor_forces
        self.external_wrench[:, self.cfg.application_mask, 3:] = motor_torques
        
        drag_wrench = compute_drag_contributions(
            linvel=self._robot.data.root_lin_vel_b,
            angvel=self._robot.data.root_ang_vel_b,
            k_lin_lin=self.body_vel_linear_damping_coefficient,
            k_lin_quad=self.body_vel_quadratic_damping_coefficient,
            k_ang_lin=self.angvel_linear_damping_coefficient,
            k_ang_quad=self.angvel_quadratic_damping_coefficient
        )
        self.external_wrench[:, 0] += drag_wrench

        # self.sim_env.robot_manager.robot.apply_disturbance()
        if self.cfg.enable_disturbance:
            max_disturb = torch.tensor(self.cfg.max_force_and_torque_disturbance).repeat(self.num_envs, 1)
            disturb_occurence = torch.bernoulli(self.cfg.disturbance_probability * torch.ones((self.num_envs), device=self.device))
            self.external_wrench[:, 0] += torch_rand_float_tensor(-max_disturb, max_disturb) * disturb_occurence.unsqueeze(1)


    def _apply_action(self):
        self._robot.set_external_force_and_torque(self.external_wrench[..., :3], self.external_wrench[..., 3:6])

    def _get_observations(self) -> dict:
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._commands - self._robot.data.root_pos_w,
                    self._robot.data.root_quat_w,
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._reward_buf.view(self.num_envs, 1),
                    self._truncated.float().view(self.num_envs, 1),
                    (self.episode_length_buf >= self.max_episode_length - 1).float().view(self.num_envs, 1)
                )
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        robot_vehicle_orientation = math_utils.yaw_quat(self._robot.data.root_quat_w)
        self._reward_buf[:] = compute_reward(
            pos_error = math_utils.quat_apply_inverse(robot_vehicle_orientation, (self._commands - self._robot.data.root_pos_w)),
            robot_quats = self._robot.data.root_quat_w,
            robot_angvels = self._robot.data.root_ang_vel_w,
            crashes = self._truncated,
            curriculum_level_multiplier = 1.0
        )
        return self._reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_vehicle_orientation = math_utils.yaw_quat(self._robot.data.root_quat_w)
        pos_error = math_utils.quat_apply_inverse(robot_vehicle_orientation, (self._commands - self._robot.data.root_pos_w))
        dist = torch.norm(pos_error, dim=1)
        self._terminated[:] = torch.where(dist > 8.0, 1, self._truncated)
        self._truncated[:] = self.episode_length_buf >= self.max_episode_length - 1
        return self._terminated.clone(), self._truncated.clone()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self._commands[env_ids] = 0


@torch.jit.script
def quat_axis(q: torch.Tensor, axis: int):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return math_utils.quat_apply(q, basis_vec)


@torch.jit.script
def torch_rand_float_tensor(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (upper - lower) * torch.rand_like(upper) + lower


@torch.jit.script
def exp_func(x: torch.Tensor, gain: float, exp: float) -> torch.Tensor:
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x: torch.Tensor, gain: float, exp: float) -> torch.Tensor:
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def compute_reward(
    pos_error: torch.Tensor,
    robot_quats: torch.Tensor,
    robot_angvels: torch.Tensor,
    crashes: torch.Tensor,
    curriculum_level_multiplier: float,
) -> torch.Tensor:    
    dist = torch.norm(pos_error, dim=1)

    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)

    dist_reward = (20 - dist) / 40.0  

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)
    
    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3
    
    total_reward = pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    total_reward[:] = curriculum_level_multiplier * total_reward
    
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)
    
    return total_reward


@torch.jit.script
def compute_drag_contributions(
    linvel: torch.Tensor,
    angvel: torch.Tensor,
    k_lin_lin: torch.Tensor,
    k_lin_quad: torch.Tensor,
    k_ang_lin: torch.Tensor,
    k_ang_quad: torch.Tensor
) -> torch.Tensor:
    # linear drag:  -k1 * linv  - k2 * ||linv|| * linv
    speed = torch.linalg.vector_norm(linvel, dim=-1, keepdim=True)
    f_lin  = -k_lin_lin  * linvel
    f_quad = -k_lin_quad * speed * linvel
    add_force = f_lin + f_quad

    # angular drag: -k1 * angv  - k2 * |ang| * angv
    t_lin  = -k_ang_lin * angvel
    t_quad = -k_ang_quad * angvel.abs() * angvel
    add_torque = t_lin + t_quad

    return torch.cat((add_force, add_torque), dim=1)