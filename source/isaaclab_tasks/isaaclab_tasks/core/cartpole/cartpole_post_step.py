# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp


@wp.func
def _wrap_to_pi(angle: wp.float32) -> wp.float32:
    """Wrap an angle [rad] to the range ``[-pi, pi]`` using Torch remainder semantics."""
    two_pi = 2.0 * wp.pi
    wrapped_angle = angle + wp.pi
    wrapped_angle = wrapped_angle - wp.floor(wrapped_angle / two_pi) * two_pi
    return wp.where((wrapped_angle == 0.0) and (angle > 0.0), wp.pi, wrapped_angle - wp.pi)


@wp.func
def _write_observation(
    env_id: int,
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    default_joint_position: wp.array2d(dtype=wp.float32),
    default_joint_velocity: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    observation: wp.array2d(dtype=wp.float32),
):
    observation[env_id, 0] = joint_position[env_id, cart_dof_idx] - default_joint_position[env_id, cart_dof_idx]
    observation[env_id, 1] = joint_position[env_id, pole_dof_idx] - default_joint_position[env_id, pole_dof_idx]
    observation[env_id, 2] = joint_velocity[env_id, cart_dof_idx] - default_joint_velocity[env_id, cart_dof_idx]
    observation[env_id, 3] = joint_velocity[env_id, pole_dof_idx] - default_joint_velocity[env_id, pole_dof_idx]


@wp.kernel
def _compute_cartpole_post_step(
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    default_joint_position: wp.array2d(dtype=wp.float32),
    default_joint_velocity: wp.array2d(dtype=wp.float32),
    episode_length: wp.array(dtype=wp.int64),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    max_episode_length: wp.int64,
    max_cart_position: wp.float32,
    reward_scale_alive: wp.float32,
    reward_scale_terminated: wp.float32,
    reward_scale_pole_position: wp.float32,
    reward_scale_cart_velocity: wp.float32,
    reward_scale_pole_velocity: wp.float32,
    step_dt: wp.float32,
    terminated: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reward: wp.array(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    cart_position = joint_position[env_id, cart_dof_idx]
    pole_position = _wrap_to_pi(joint_position[env_id, pole_dof_idx])
    cart_velocity = joint_velocity[env_id, cart_dof_idx]
    pole_velocity = joint_velocity[env_id, pole_dof_idx]

    is_terminated = wp.abs(cart_position) > max_cart_position
    terminated[env_id] = is_terminated
    time_out[env_id] = episode_length[env_id] >= max_episode_length

    alive_reward = reward_scale_alive
    termination_reward = wp.float32(0.0)
    if is_terminated:
        alive_reward = wp.float32(0.0)
        termination_reward = reward_scale_terminated
    pole_position_reward = reward_scale_pole_position * pole_position * pole_position
    cart_velocity_reward = reward_scale_cart_velocity * wp.abs(cart_velocity)
    pole_velocity_reward = reward_scale_pole_velocity * wp.abs(pole_velocity)
    reward[env_id] = (
        alive_reward + termination_reward + pole_position_reward + cart_velocity_reward + pole_velocity_reward
    ) * step_dt

    _write_observation(
        env_id,
        joint_position,
        joint_velocity,
        default_joint_position,
        default_joint_velocity,
        cart_dof_idx,
        pole_dof_idx,
        observation,
    )


@wp.kernel
def _compute_cartpole_observation(
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    default_joint_position: wp.array2d(dtype=wp.float32),
    default_joint_velocity: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    observation: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    _write_observation(
        env_id,
        joint_position,
        joint_velocity,
        default_joint_position,
        default_joint_velocity,
        cart_dof_idx,
        pole_dof_idx,
        observation,
    )


class _CartpolePostStepBuffers:
    """Persistent task outputs used by the fused Cartpole post-step kernels."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.terminated = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.time_out = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.reward = wp.zeros(num_envs, dtype=wp.float32, device=device)
        # RL runners may retain the current observation until the next step completes.
        self._observations = (
            wp.zeros((num_envs, 4), dtype=wp.float32, device=device),
            wp.zeros((num_envs, 4), dtype=wp.float32, device=device),
        )
        self._observation_torch = tuple(wp.to_torch(observation) for observation in self._observations)
        self._observation_index = 0
        self.observation = self._observations[self._observation_index]
        self.observation_torch = self._observation_torch[self._observation_index]
        self.terminated_torch = wp.to_torch(self.terminated)
        self.time_out_torch = wp.to_torch(self.time_out)
        self.reward_torch = wp.to_torch(self.reward)

    def compute_post_step(self, env) -> None:
        """Compute Cartpole dones, rewards, and observations in one launch."""
        self._observation_index ^= 1
        self.observation = self._observations[self._observation_index]
        self.observation_torch = self._observation_torch[self._observation_index]
        wp.launch(
            _compute_cartpole_post_step,
            dim=self.num_envs,
            inputs=[
                env._joint_position_warp,
                env._joint_velocity_warp,
                env.cartpole.data.default_joint_pos.warp,
                env.cartpole.data.default_joint_vel.warp,
                env.episode_length_buf,
                env._cart_dof_idx[0],
                env._pole_dof_idx[0],
                env.max_episode_length,
                env.cfg.max_cart_pos,
                env.cfg.rew_scale_alive,
                env.cfg.rew_scale_terminated,
                env.cfg.rew_scale_pole_pos,
                env.cfg.rew_scale_cart_vel,
                env.cfg.rew_scale_pole_vel,
                env.step_dt,
            ],
            outputs=[self.terminated, self.time_out, self.reward, self.observation],
            device=self.device,
        )

    def compute_observation(self, env) -> None:
        """Refresh observations after an index-based reset."""
        wp.launch(
            _compute_cartpole_observation,
            dim=self.num_envs,
            inputs=[
                env._joint_position_warp,
                env._joint_velocity_warp,
                env.cartpole.data.default_joint_pos.warp,
                env.cartpole.data.default_joint_vel.warp,
                env._cart_dof_idx[0],
                env._pole_dof_idx[0],
            ],
            outputs=[self.observation],
            device=self.device,
        )
