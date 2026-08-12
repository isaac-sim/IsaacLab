# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from isaaclab_tasks.core.pendulum.pendulum_marl_env_cfg import PendulumMARLEnvCfg


class PendulumMARLEnv(DirectMARLEnv):
    """Multi-agent cart-double-pendulum balancing environment.

    Two agents (``cart`` and ``pendulum``) cooperate to keep the double pendulum upright. They
    share common reward and termination signals while observing independently.
    """

    cfg: PendulumMARLEnvCfg

    def __init__(self, cfg: PendulumMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)

        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch
        self._success_required_steps = round(self.cfg.success_duration_s / self.step_dt)
        self._consecutive_upright_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        # PhysX replication requires explicit collision filtering between environments.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(
            target=self.actions["cart"] * self.cfg.cart_action_scale, joint_ids=self._cart_dof_idx
        )
        self.robot.set_joint_effort_target_index(
            target=self.actions["pendulum"] * self.cfg.pendulum_action_scale, joint_ids=self._pendulum_dof_idx
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
        team_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_pendulum_pos,
            self.cfg.rew_scale_pendulum_vel,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_action,
            self.joint_vel[:, self._cart_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]),
            self.joint_vel[:, self._pole_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]),
            self.joint_vel[:, self._pendulum_dof_idx[0]],
            links_upright(
                self.joint_pos[:, self._pole_dof_idx[0]],
                self.joint_pos[:, self._pendulum_dof_idx[0]],
                self.cfg.success_upright_angle,
            ),
            self.actions["cart"],
            self.actions["pendulum"],
            math.prod(self.terminated_dict.values()),
            self.step_dt,
        )
        return {agent: team_reward for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        upright = links_upright(
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._pendulum_dof_idx[0]],
            self.cfg.success_upright_angle,
        )
        self._consecutive_upright_steps = update_upright_steps(self._consecutive_upright_steps, upright)
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, "_consecutive_upright_steps") and hasattr(self, "time_out_dict"):
            agent = self.cfg.possible_agents[0]
            success = compute_success(
                self.time_out_dict[agent][env_ids],
                self.terminated_dict[agent][env_ids],
                self._consecutive_upright_steps[env_ids],
                self._success_required_steps,
            )
            self.extras.setdefault("log", {})["Metrics/success_rate"] = success.float().mean().item()
            self._consecutive_upright_steps[env_ids] = 0
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids]
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
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids]

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids]
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


@torch.jit.script
def normalize_angle(angle):
    """Wrap an angle [rad] to the range ``[-pi, pi]``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def links_upright(
    pole_pos: torch.Tensor,
    pendulum_pos: torch.Tensor,
    max_angle: float,
) -> torch.Tensor:
    """Return whether both physical links are within the upright angle limit.

    Args:
        pole_pos: Upper-link joint angle [rad], shape ``(num_envs,)``, dtype ``torch.float``.
        pendulum_pos: Lower-link angle relative to the upper link [rad], shape ``(num_envs,)``,
            dtype ``torch.float``.
        max_angle: Maximum permitted absolute world-relative link angle [rad].

    Returns:
        Whether both links are upright, shape ``(num_envs,)``, dtype ``torch.bool``.
    """
    upper_upright = normalize_angle(pole_pos).abs() <= max_angle
    lower_upright = normalize_angle(pole_pos + pendulum_pos).abs() <= max_angle
    return upper_upright & lower_upright


def update_upright_steps(upright_steps: torch.Tensor, upright: torch.Tensor) -> torch.Tensor:
    """Update per-environment consecutive upright control-step counts.

    Args:
        upright_steps: Current consecutive upright-step counts, shape ``(num_envs,)``, dtype
            ``torch.long``.
        upright: Whether each environment is upright, shape ``(num_envs,)``, dtype ``torch.bool``.

    Returns:
        Updated consecutive upright-step counts, shape ``(num_envs,)``, dtype ``torch.long``.
    """
    return torch.where(upright, upright_steps + 1, torch.zeros_like(upright_steps))


def compute_success(
    time_out: torch.Tensor,
    terminated: torch.Tensor,
    upright_steps: torch.Tensor,
    required_steps: int,
) -> torch.Tensor:
    """Return successful episodes from timeout, failure, and upright-window state.

    Args:
        time_out: Whether each episode reached its time limit, shape ``(num_envs,)``, dtype
            ``torch.bool``.
        terminated: Whether each episode terminated before its time limit, shape ``(num_envs,)``,
            dtype ``torch.bool``.
        upright_steps: Consecutive upright-step counts, shape ``(num_envs,)``, dtype ``torch.long``.
        required_steps: Minimum consecutive upright-step count required for success.

    Returns:
        Whether each episode succeeded, shape ``(num_envs,)``, dtype ``torch.bool``.
    """
    return time_out & ~terminated & (upright_steps >= required_steps)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    rew_scale_upright: float,
    rew_scale_action: float,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    upright: torch.Tensor,
    cart_action: torch.Tensor,
    pendulum_action: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    lower_angle = normalize_angle(pole_pos + pendulum_pos)
    lower_velocity = pole_vel + pendulum_vel
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.cos(pole_pos)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.cos(lower_angle)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.abs(lower_velocity)
    rew_upright = rew_scale_upright * upright.float()
    rew_action = rew_scale_action * (
        torch.sum(torch.square(cart_action), dim=1) + torch.sum(torch.square(pendulum_action), dim=1)
    )
    return (
        rew_alive
        + rew_termination
        + rew_pole_pos
        + rew_pendulum_pos
        + rew_cart_vel
        + rew_pole_vel
        + rew_pendulum_vel
        + rew_upright
        + rew_action
    ) * step_dt
