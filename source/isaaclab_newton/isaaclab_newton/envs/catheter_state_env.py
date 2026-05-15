# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Catheter navigation environment with state-based observations.

Uses the XPBD Cosserat rod solver with proximal kinematic control
(push / rotate at the catheter root) and flat vector observations.
No Isaac Sim / Omniverse dependency — pure PyTorch + Warp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch


@dataclass
class CatheterStateEnvCfg:
    """Configuration for :class:`CatheterStateEnv`."""

    # --- environment ---
    num_envs: int = 512
    device: str = "cuda"
    max_episode_steps: int = 1000
    dt: float = 1.0 / 60.0

    # --- rod geometry / material ---
    num_segments: int = 20
    rod_length: float = 0.2
    rod_radius: float = 0.00045
    young_modulus: float = 1e8
    density: float = 6450.0
    bend_stiffness: float = 0.1
    twist_stiffness: float = 0.4
    damping: float = 0.01
    num_substeps: int = 2

    # --- action ---
    action_scale_push: float = 0.01       # m/s
    action_scale_rotate: float = 1.0      # rad/s

    # --- target / reward ---
    target_region_center: tuple[float, float, float] = (0.15, 0.0, 0.0)
    target_region_radius: float = 0.05
    target_reached_threshold: float = 0.005
    rew_distance_scale: float = -10.0
    rew_time_penalty: float = -0.01
    rew_reached_bonus: float = 100.0


class CatheterStateEnv(gym.Env):
    """Vectorised catheter navigation env with state-based observations.

    **Action space** ``Box(2)``:
        ``[push_velocity, rotate_velocity]`` normalised to [-1, 1],
        scaled by ``action_scale_push`` and ``action_scale_rotate``.

    **Observation space** ``Box(obs_dim)``:
        Flat concatenation of: segment positions ``(N*3)``, tip position
        ``(3)``, tip velocity ``(3)``, target position ``(3)``, root-x
        insertion depth ``(1)``.

    The environment is internally vectorised: a single ``RodSolver``
    instance with ``num_envs`` environments runs on GPU.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, cfg: CatheterStateEnvCfg | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg or CatheterStateEnvCfg(**kwargs)
        c = self.cfg
        self.device = c.device
        self.num_envs = c.num_envs
        self.max_episode_length = c.max_episode_steps

        from isaaclab_newton.solvers import RodConfig, RodSolver

        rod_cfg = RodConfig()
        rod_cfg.geometry.num_segments = c.num_segments
        rod_cfg.geometry.rest_length = c.rod_length
        rod_cfg.geometry.segment_length = c.rod_length / c.num_segments
        rod_cfg.geometry.radius = c.rod_radius
        rod_cfg.material.young_modulus = c.young_modulus
        rod_cfg.material.density = c.density
        rod_cfg.material.bend_stiffness = c.bend_stiffness
        rod_cfg.material.twist_stiffness = c.twist_stiffness
        rod_cfg.material.damping = c.damping
        rod_cfg.solver.dt = c.dt
        rod_cfg.solver.num_substeps = c.num_substeps
        rod_cfg.solver.gravity = (0.0, 0.0, -9.81)
        rod_cfg.device = c.device

        self.solver = RodSolver(rod_cfg, num_envs=c.num_envs, device=c.device)
        self.solver.data.fix_segment(slice(None), 0)

        n_seg = c.num_segments
        self._obs_dim = n_seg * 3 + 3 + 3 + 3 + 1

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self.target_positions = torch.zeros(
            (c.num_envs, 3), device=c.device, dtype=torch.float32,
        )
        self.episode_length_buf = torch.zeros(
            c.num_envs, device=c.device, dtype=torch.long,
        )
        self._initial_root_x = torch.zeros(
            c.num_envs, device=c.device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(env_ids)
        obs = self._compute_obs()
        return obs, {}

    def step(self, actions: torch.Tensor):
        c = self.cfg

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)

        actions = actions.clamp(-1.0, 1.0)
        push_vel = actions[:, 0] * c.action_scale_push
        rot_vel = actions[:, 1] * c.action_scale_rotate

        self.solver.apply_proximal_control(push_vel, rot_vel, c.dt)
        self.solver.step(c.dt)

        self.episode_length_buf += 1

        obs = self._compute_obs()
        rewards = self._compute_rewards()
        terminated, truncated = self._compute_dones()

        resets = terminated | truncated
        if resets.any():
            reset_ids = resets.nonzero(as_tuple=False).squeeze(-1)
            self._reset_envs(reset_ids)

        return obs, rewards, terminated, truncated, {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids: torch.Tensor):
        self.solver.data.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._randomise_targets(env_ids)
        self._initial_root_x[env_ids] = self.solver.data.positions[env_ids, 0, 0]

    def _randomise_targets(self, env_ids: torch.Tensor):
        c = self.cfg
        n = len(env_ids)
        center = torch.tensor(c.target_region_center, device=self.device)
        offsets = (torch.rand(n, 3, device=self.device) - 0.5) * 2.0 * c.target_region_radius
        self.target_positions[env_ids] = center.unsqueeze(0) + offsets

    def _compute_obs(self) -> torch.Tensor:
        pos = self.solver.data.positions
        vel = self.solver.data.velocities

        tip_pos = pos[:, -1, :]
        tip_vel = vel[:, -1, :]
        insertion_depth = (pos[:, 0, 0] - self._initial_root_x).unsqueeze(-1)

        obs = torch.cat([
            pos.reshape(self.num_envs, -1),
            tip_pos,
            tip_vel,
            self.target_positions,
            insertion_depth,
        ], dim=-1)
        return obs

    def _compute_rewards(self) -> torch.Tensor:
        c = self.cfg
        tip_pos = self.solver.data.positions[:, -1, :]
        dist = torch.norm(tip_pos - self.target_positions, dim=-1)

        reward = (
            c.rew_distance_scale * dist
            + c.rew_time_penalty
            + c.rew_reached_bonus * (dist < c.target_reached_threshold).float()
        )
        return reward

    def _compute_dones(self):
        tip_pos = self.solver.data.positions[:, -1, :]
        dist = torch.norm(tip_pos - self.target_positions, dim=-1)

        terminated = dist < self.cfg.target_reached_threshold
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ------------------------------------------------------------------
    # properties expected by rsl_rl / Isaac Lab wrappers
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        return self

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._episode_length_buf = value
