# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab_experimental.utils.warp.utils import wrap_to_pi

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

if TYPE_CHECKING:
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg


@wp.kernel
def get_observations(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    observations: wp.array(dtype=wp.vec4f),
):
    """Write per-env cart and pole joint position/velocity deltas from defaults into the observation vec4."""
    env_index = wp.tid()
    observations[env_index][0] = joint_pos[env_index, cart_dof_idx] - default_joint_pos[env_index, cart_dof_idx]
    observations[env_index][1] = joint_pos[env_index, pole_dof_idx] - default_joint_pos[env_index, pole_dof_idx]
    observations[env_index][2] = joint_vel[env_index, cart_dof_idx] - default_joint_vel[env_index, cart_dof_idx]
    observations[env_index][3] = joint_vel[env_index, pole_dof_idx] - default_joint_vel[env_index, pole_dof_idx]


@wp.kernel
def update_actions(
    input_actions: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    action_scale: wp.float32,
    cart_dof_idx: wp.int32,
):
    """Scale the input action and write it into the cart DOF slot of the joint-effort action buffer."""
    env_index = wp.tid()
    actions[env_index, cart_dof_idx] = action_scale * input_actions[env_index, 0]


@wp.kernel
def get_dones(
    joint_pos: wp.array2d(dtype=wp.float32),
    episode_length_buf: wp.array(dtype=wp.int32),
    cart_dof_idx: wp.int32,
    max_episode_length: wp.int32,
    max_cart_pos: wp.float32,
    out_of_bounds: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    """Flag per-env cart-out-of-bounds and time-out termination, and their union into the reset buffer."""
    env_index = wp.tid()
    out_of_bounds[env_index] = wp.abs(joint_pos[env_index, cart_dof_idx]) > max_cart_pos
    time_out[env_index] = episode_length_buf[env_index] >= max_episode_length
    reset[env_index] = out_of_bounds[env_index] or time_out[env_index]


@wp.func
def compute_rew_alive(rew_scale_alive: wp.float32, reset_terminated: bool) -> wp.float32:
    if reset_terminated:
        return wp.float32(0.0)
    return rew_scale_alive


@wp.func
def compute_rew_termination(rew_scale_terminated: wp.float32, reset_terminated: bool) -> wp.float32:
    if reset_terminated:
        return rew_scale_terminated
    return wp.float32(0.0)


@wp.func
def compute_rew_pole_pos(
    rew_scale_pole_pos: wp.float32,
    pole_pos: wp.float32,
) -> wp.float32:
    pole_pos = wrap_to_pi(pole_pos)
    return rew_scale_pole_pos * pole_pos * pole_pos


@wp.func
def compute_rew_cart_vel(
    rew_scale_cart_vel: wp.float32,
    cart_vel: wp.float32,
) -> wp.float32:
    return rew_scale_cart_vel * wp.abs(cart_vel)


@wp.func
def compute_rew_pole_vel(
    rew_scale_pole_vel: wp.float32,
    pole_vel: wp.float32,
) -> wp.float32:
    return rew_scale_pole_vel * wp.abs(pole_vel)


@wp.kernel
def compute_rewards(
    rew_scale_alive: wp.float32,
    rew_scale_terminated: wp.float32,
    rew_scale_pole_pos: wp.float32,
    rew_scale_cart_vel: wp.float32,
    rew_scale_pole_vel: wp.float32,
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    reset_terminated: wp.array(dtype=wp.bool),
    step_dt: wp.float32,
    reward: wp.array(dtype=wp.float32),
):
    """Accumulate the scaled alive, termination, and pole/cart position/velocity reward terms."""
    env_index = wp.tid()
    reward[env_index] = step_dt * (
        compute_rew_alive(rew_scale_alive, reset_terminated[env_index])
        + compute_rew_termination(rew_scale_terminated, reset_terminated[env_index])
        + compute_rew_pole_pos(rew_scale_pole_pos, joint_pos[env_index, pole_dof_idx])
        + compute_rew_cart_vel(rew_scale_cart_vel, joint_vel[env_index, cart_dof_idx])
        + compute_rew_pole_vel(rew_scale_pole_vel, joint_vel[env_index, pole_dof_idx])
    )


@wp.kernel
def reset(
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_vel_limits: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    initial_cart_position_range: wp.vec2f,
    initial_cart_velocity_range: wp.vec2f,
    initial_pole_angle_range: wp.vec2f,
    initial_pole_velocity_range: wp.vec2f,
    env_mask: wp.array(dtype=wp.bool),
    state: wp.array(dtype=wp.uint32),
):
    """Sample randomized cart/pole joint positions and velocities within soft limits for masked envs."""
    env_index = wp.tid()
    if env_mask[env_index]:
        rng_state = state[env_index]
        cart_pos = default_joint_pos[env_index, cart_dof_idx] + wp.randf(
            rng_state, initial_cart_position_range[0], initial_cart_position_range[1]
        )
        pole_pos = default_joint_pos[env_index, pole_dof_idx] + wp.randf(
            rng_state, initial_pole_angle_range[0], initial_pole_angle_range[1]
        )
        cart_vel = default_joint_vel[env_index, cart_dof_idx] + wp.randf(
            rng_state, initial_cart_velocity_range[0], initial_cart_velocity_range[1]
        )
        pole_vel = default_joint_vel[env_index, pole_dof_idx] + wp.randf(
            rng_state, initial_pole_velocity_range[0], initial_pole_velocity_range[1]
        )

        cart_pos_limits = soft_joint_pos_limits[env_index, cart_dof_idx]
        pole_pos_limits = soft_joint_pos_limits[env_index, pole_dof_idx]
        cart_vel_limit = soft_joint_vel_limits[env_index, cart_dof_idx]
        pole_vel_limit = soft_joint_vel_limits[env_index, pole_dof_idx]

        joint_pos[env_index, cart_dof_idx] = wp.clamp(cart_pos, cart_pos_limits[0], cart_pos_limits[1])
        joint_pos[env_index, pole_dof_idx] = wp.clamp(pole_pos, pole_pos_limits[0], pole_pos_limits[1])
        joint_vel[env_index, cart_dof_idx] = wp.clamp(cart_vel, -cart_vel_limit, cart_vel_limit)
        joint_vel[env_index, pole_dof_idx] = wp.clamp(pole_vel, -pole_vel_limit, pole_vel_limit)
        state[env_index] = rng_state


@wp.kernel
def initialize_state(
    state: wp.array(dtype=wp.uint32),
    seed: wp.int32,
):
    """Initialize each env's random number generator state from the seed."""
    env_index = wp.tid()
    state[env_index] = wp.rand_init(seed, env_index)


class CartpoleWarpEnv(DirectRLEnvWarp):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # Get the indices (develop API: find_joints returns (indices, names))
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)

        self.action_scale = self.cfg.action_scale

        # Simulation bindings
        # Note: these are direct memory views into the Newton simulation data, they should not be modified directly
        self.joint_pos = self.cartpole.data.joint_pos.warp
        self.joint_vel = self.cartpole.data.joint_vel.warp

        # Buffers
        self.observations = wp.zeros((self.num_envs), dtype=wp.vec4f, device=self.device)
        self.actions = wp.zeros((self.num_envs, self.cartpole.num_joints), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros((self.num_envs), dtype=wp.float32, device=self.device)
        self.states = wp.zeros((self.num_envs), dtype=wp.uint32, device=self.device)

        if self.cfg.seed is None:
            self.cfg.seed = -1

        wp.launch(
            initialize_state,
            dim=self.num_envs,
            inputs=[
                self.states,
                self.cfg.seed,
            ],
        )

        # Bind torch buffers to warp buffers
        self.torch_obs_buf = wp.to_torch(self.observations)
        self.torch_reward_buf = wp.to_torch(self.rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self.torch_episode_length_buf = self.episode_length_buf  # already a torch tensor via wp.to_torch

    def _setup_scene(self) -> None:
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_orientation = (-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
        light_cfg.func("/World/Light", light_cfg, orientation=light_orientation)

    def _pre_physics_step(self, actions: wp.array) -> None:
        wp.launch(
            update_actions,
            dim=self.num_envs,
            inputs=[
                actions,
                self.actions,
                self.action_scale,
                self._cart_dof_idx[0],
            ],
        )

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target_mask(target=self.actions)

    def _get_observations(self) -> dict:
        wp.launch(
            get_observations,
            dim=self.num_envs,
            inputs=[
                self.joint_pos,
                self.joint_vel,
                self.cartpole.data.default_joint_pos.warp,
                self.cartpole.data.default_joint_vel.warp,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.observations,
            ],
        )
        return {"policy": self.torch_obs_buf}

    def _get_rewards(self) -> None:
        wp.launch(
            compute_rewards,
            dim=self.num_envs,
            inputs=[
                self.cfg.rew_scale_alive,
                self.cfg.rew_scale_terminated,
                self.cfg.rew_scale_pole_pos,
                self.cfg.rew_scale_cart_vel,
                self.cfg.rew_scale_pole_vel,
                self.joint_pos,
                self.joint_vel,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.reset_terminated,
                self.step_dt,
                self.rewards,
            ],
        )

    def _get_dones(self) -> None:
        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.joint_pos,
                self._episode_length_buf_wp,
                self._cart_dof_idx[0],
                self.max_episode_length,
                self.cfg.max_cart_pos,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
        )

    def _reset_idx(self, mask: wp.array | None = None) -> None:
        if mask is None:
            mask = self._ALL_ENV_MASK

        super()._reset_idx(mask)

        wp.launch(
            reset,
            dim=self.num_envs,
            inputs=[
                self.cartpole.data.default_joint_pos.warp,
                self.cartpole.data.default_joint_vel.warp,
                self.joint_pos,
                self.joint_vel,
                self.cartpole.data.soft_joint_pos_limits.warp,
                self.cartpole.data.soft_joint_vel_limits.warp,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.cfg.initial_cart_position_range,
                self.cfg.initial_cart_velocity_range,
                self.cfg.initial_pole_angle_range,
                self.cfg.initial_pole_velocity_range,
                mask,
                self.states,
            ],
        )
