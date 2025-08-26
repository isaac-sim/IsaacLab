# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers

from .controller_cfg import LeeControllerCfg
from .drone_env_cfg import DroneEnvCfg
from .thruster_cfg import ThrusterCfg
from .utils import rand_range

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class DroneEnv(DirectRLEnv):
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._commands_w = torch.zeros((self.num_envs, 3), device=self.device)

        # mdp buffers
        self.rew_keys = ["position_tracking", "distance_rew", "behavior_shaping", "early_termination"]
        self._episode_reward_sums = torch.zeros((self.num_envs, len(self.rew_keys)), device=self.device)
        self._reward_term_buf = torch.zeros((self.num_envs, len(self.rew_keys)), device=self.device)
        self._reward_buf = torch.zeros((self.num_envs,), device=self.device)
        self._terminated = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._truncated = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # drag
        self.body_vel_linear_damping_coef = torch.tensor(self.cfg.body_vel_linear_damping_coef).to(self.device)
        self.body_vel_quadratic_damping_coef = torch.tensor(self.cfg.body_vel_quadratic_damping_coef).to(self.device)
        self.angvel_linear_damping_coef = torch.tensor(self.cfg.angvel_linear_damping_coef).to(self.device)
        self.angvel_quadratic_damping_coef = torch.tensor(self.cfg.angvel_quadratic_damping_coef).to(self.device)

        self.set_debug_vis(self.cfg.debug_vis)

        # logging
        self.extras["log"] = {f"Episode_Reward/{key}": 0.0 for key in self.rew_keys}

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = actions.clamp(-10, 10)
        controller_output = self.controller(self._processed_actions)

        # call actuator model to get forces and torque
        ref_motor_thrusts = torch.bmm(self.inv_wrench_allocation_matrix, controller_output.unsqueeze(-1)).squeeze(-1)
        motor_thrusts = self.thruster.update_motor_thrusts(ref_motor_thrusts)
        zero_thrust = torch.zeros_like(motor_thrusts)
        motor_forces_b = torch.stack((zero_thrust, zero_thrust, motor_thrusts), dim=2)
        motor_torques_b = self.thrust_to_torque_ratio * motor_forces_b * (-self.motor_directions[None, :, None])

        self.external_wrench_b = torch.zeros((self.num_envs, self._robot.num_bodies, 6), device=self.device)
        self.external_wrench_b[:, self.cfg.application_mask, :3] = motor_forces_b
        self.external_wrench_b[:, self.cfg.application_mask, 3:] = motor_torques_b

        drag_wrench = compute_drag_contributions(
            linvel=self._robot.data.root_lin_vel_b,
            angvel=self._robot.data.root_ang_vel_b,
            k_lin_lin=self.body_vel_linear_damping_coef,
            k_lin_quad=self.body_vel_quadratic_damping_coef,
            k_ang_lin=self.angvel_linear_damping_coef,
            k_ang_quad=self.angvel_quadratic_damping_coef,
        )
        self.external_wrench_b[:, 0] += drag_wrench

        if self.cfg.enable_disturbance:
            max_disturb = torch.tensor(self.cfg.max_wrench_disturbance, device=self.device).repeat(self.num_envs, 1)
            disturb_occur = torch.bernoulli(self.cfg.disturb_prob * torch.ones((self.num_envs), device=self.device))
            self.external_wrench_b[:, 0] += rand_range(-max_disturb, max_disturb) * disturb_occur.unsqueeze(1)

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self.external_wrench_b[..., :3], self.external_wrench_b[..., 3:6], is_global=False  # very important
        )

    def _get_observations(self) -> dict:
        pos_error_w = self._commands_w - self._robot.data.root_pos_w
        obs = torch.cat(
            [
                math_utils.quat_apply_inverse(self._robot.data.root_quat_w, (pos_error_w)),
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        robot_vehicle_orientation = math_utils.yaw_quat(self._robot.data.root_quat_w)
        pos_error = self._commands_w - self._robot.data.root_pos_w
        self._reward_term_buf[:, :3] = compute_reward_components(
            pos_error=math_utils.quat_apply_inverse(robot_vehicle_orientation, pos_error),
            robot_quats=self._robot.data.root_quat_w,
            robot_angvels=self._robot.data.root_ang_vel_w,
        )
        self._reward_term_buf[:, 3] = self._terminated.float() * -20
        self._episode_reward_sums += self._reward_term_buf
        return torch.sum(self._reward_term_buf, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._terminated[:] = torch.norm(self._commands_w - self._robot.data.root_pos_w, dim=1) > 8.0
        self._truncated[:] = self.episode_length_buf >= self.max_episode_length - 1
        return self._terminated.clone(), self._truncated.clone()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._commands_w[env_ids] = torch.zeros((len(env_ids), 3), device=self.device) + self.scene.env_origins[env_ids]
        if self._sim_step_counter == 0:
            # Setup Controller at first reset
            controller_cfg = LeeControllerCfg()
            self.controller = controller_cfg.class_type(controller_cfg, self)
            self.motor_directions = torch.tensor(self.cfg.motor_directions, device=self.device)

            # Setup Thruster at first reset
            thruster_cfg = ThrusterCfg(dt=self.cfg.sim.dt)
            self.thruster = thruster_cfg.class_type(self.num_envs, cfg=thruster_cfg, device=self.device)
            self.thrust_to_torque_ratio = self.cfg.thrust_to_torque_ratio

            # randomly initialize first episode length so env episode progress spread out.
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

            wrench_alloc_matrix = torch.tensor(self.cfg.allocation_matrix, device=self.device)
            self.inv_wrench_allocation_matrix = torch.linalg.pinv(wrench_alloc_matrix).expand(self.num_envs, -1, -1)

        self.thruster.reset_idx(env_ids)

        rew_sum_avg = torch.mean(self._episode_reward_sums, dim=0) / self.max_episode_length_s
        for i, key in enumerate(self.rew_keys):
            self.extras["log"][f"Episode_Reward/{key}"] = rew_sum_avg[i].item()
        self._episode_reward_sums[env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._commands_w)


@torch.jit.script
def quat_axis(q: torch.Tensor, axis: int):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return math_utils.quat_apply(q, basis_vec)


@torch.jit.script
def exp_func(x: torch.Tensor, gain: float, exp: float) -> torch.Tensor:
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x: torch.Tensor, gain: float, exp: float) -> torch.Tensor:
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def compute_reward_components(
    pos_error: torch.Tensor,
    robot_quats: torch.Tensor,
    robot_angvels: torch.Tensor,
) -> torch.Tensor:
    """
    Returns an (N, 3) tensor with:
      [:, 0] -> pos_reward
      [:, 1] -> dist_reward
      [:, 2] -> pos_reward * (up_reward + ang_vel_reward)
    """
    # distances
    dist = torch.norm(pos_error, dim=1)

    # the original terms
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0)
    dist_reward = (20.0 - dist) / 40.0

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1.0 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3.0

    posture_term = pos_reward * (up_reward + ang_vel_reward)

    return torch.stack((pos_reward, dist_reward, posture_term), dim=1)


@torch.jit.script
def compute_drag_contributions(
    linvel: torch.Tensor,
    angvel: torch.Tensor,
    k_lin_lin: torch.Tensor,
    k_lin_quad: torch.Tensor,
    k_ang_lin: torch.Tensor,
    k_ang_quad: torch.Tensor,
) -> torch.Tensor:
    # linear drag:  -k1 * linv  - k2 * ||linv|| * linv
    speed = torch.linalg.vector_norm(linvel, dim=-1, keepdim=True)
    f_lin = -k_lin_lin * linvel
    f_quad = -k_lin_quad * speed * linvel
    add_force = f_lin + f_quad

    # angular drag: -k1 * angv  - k2 * |ang| * angv
    t_lin = -k_ang_lin * angvel
    t_quad = -k_ang_quad * angvel.abs() * angvel
    add_torque = t_lin + t_quad

    return torch.cat((add_force, add_torque), dim=1)
