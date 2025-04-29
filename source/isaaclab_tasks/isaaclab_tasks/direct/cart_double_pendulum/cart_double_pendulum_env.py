# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class CartDoublePendulumEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    possible_agents = ["cart", "pendulum"]
    action_spaces = {"cart": 1, "pendulum": 1}
    observation_spaces = {"cart": 4, "pendulum": 3}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]
    initial_pendulum_angle_range = [-0.25, 0.25]  # the range in which the pendulum angle is sampled from on reset [rad]

    # action scales
    cart_action_scale = 100.0  # [N]
    pendulum_action_scale = 50.0  # [Nm]

    # reward scales
    eps_alive = 1.0
    eps_terminated = -2.0
    eps_cart_pos = 0
    eps_cart_vel = -0.01
    eps_pole_pos = -1.0
    eps_pole_vel = -0.01
    eps_pendulum_pos = -1.0
    eps_pendulum_vel = -0.01

class CartDoublePendulumEnv(DirectMARLEnv):
    cfg: CartDoublePendulumEnvCfg

    def __init__(self, cfg: CartDoublePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions["cart"] * self.cfg.cart_action_scale, joint_ids=self._cart_dof_idx
        )
        self.robot.set_joint_effort_target(
            self.actions["pendulum"] * self.cfg.pendulum_action_scale, joint_ids=self._pendulum_dof_idx
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pole_joint_pos = normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1))
        pendulum_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1))
        observations = {
            "cart": torch.cat(
                (
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    pole_joint_pos,
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
            "pendulum": torch.cat(
                (
                    pole_joint_pos + pendulum_joint_pos,
                    pendulum_joint_pos,
                    self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        P_cart_0, P_pendulum_0, Delta_P_cart, Delta_P_pendulum, total_reward = compute_rewards(
            1.0, # alpha
            1.0, # beta
            self.cfg.eps_alive, # eps_alive
            self.cfg.eps_terminated, # eps_terminated
            self.cfg.eps_cart_vel, # eps_cart_vel
            self.cfg.eps_pole_pos, # eps_pole_pos
            self.cfg.eps_pole_vel, # eps_pole_vel
            self.cfg.eps_pendulum_pos,   # eps_pendulum_pos
            self.cfg.eps_pendulum_vel,    # eps_pendulum_vel
            self.joint_vel[:, self._cart_dof_idx[0]], # cart_vel
            normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]), # pole_pos
            self.joint_vel[:, self._pole_dof_idx[0]], # pole_vel
            normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]), # pendulum_pos
            self.joint_vel[:, self._pendulum_dof_idx[0]],  # pendulum_vel
            math.prod(self.terminated_dict.values()), # reset_terminated
        )
        if "log" not in self.extras:
            self.extras["log"] = dict() 
        self.extras["log"]["P_cart_0"] = P_cart_0.mean()
        self.extras["log"]["P_pendulum_0"] = P_pendulum_0.mean()
        self.extras["log"]["Delta_P_cart"] = Delta_P_cart.mean()
        self.extras["log"]["Delta_P_pendulum"] = Delta_P_pendulum.mean()
        return total_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            self.cfg.initial_pendulum_angle_range[0] * math.pi,
            self.cfg.initial_pendulum_angle_range[1] * math.pi,
            joint_pos[:, self._pendulum_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards(
    alpha: float,
    beta: float,
    eps_alive: float,
    eps_terminated: float,
    eps_cart_vel: float,
    eps_pole_pos: float,
    eps_pole_vel: float,
    eps_pendulum_pos: float,
    eps_pendulum_vel: float,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # Base reward components
    P_cart_0 = (
        eps_alive * (1.0 - reset_terminated.float())
        + eps_terminated * reset_terminated.float()
        + eps_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    )
    
    P_pendulum_0 = (
        eps_alive * (1.0 - reset_terminated.float())
        + eps_terminated * reset_terminated.float()
    )
    
    # Cooperative (mutualistic) terms
    Delta_P_cart = (
        eps_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
        + eps_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    )
    
    Delta_P_pendulum = (
        eps_pendulum_pos * torch.sum(torch.square(pole_pos + pendulum_pos).unsqueeze(dim=1), dim=-1)
        + eps_pendulum_vel * torch.sum(torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1)
    )
    
    # Final rewards incorporating mutualistic principles
    R_cart = alpha * P_cart_0 + beta * Delta_P_cart
    R_pendulum = alpha * P_pendulum_0 + beta * Delta_P_pendulum
    
    total_reward = {"cart": R_cart, "pendulum": R_pendulum}
    
    return P_cart_0, P_pendulum_0, Delta_P_cart, Delta_P_pendulum, total_reward