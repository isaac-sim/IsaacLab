# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass


@configclass
class CartpoleWarpEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    solver_cfg = MJWarpSolverCfg(
        njmax=5,
        nconmax=3,
        ls_iterations=3,
        cone="pyramidal",
        impratio=1,
        update_data_interval=1,
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=newton_cfg)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


@wp.kernel
def get_observations(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    observations: wp.array(dtype=wp.vec4f),
):
    env_index = wp.tid()
    observations[env_index][0] = joint_pos[env_index, pole_dof_idx]
    observations[env_index][1] = joint_vel[env_index, pole_dof_idx]
    observations[env_index][2] = joint_pos[env_index, cart_dof_idx]
    observations[env_index][3] = joint_vel[env_index, cart_dof_idx]


@wp.kernel
def update_actions(
    input_actions: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    action_scale: wp.float32,
):
    env_index = wp.tid()
    actions[env_index, 0] = action_scale * input_actions[env_index, 0]


@wp.kernel
def get_dones(
    joint_pos: wp.array2d(dtype=wp.float32),
    episode_length_buf: wp.array(dtype=wp.int32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    max_episode_length: wp.int32,
    max_cart_pos: wp.float32,
    out_of_bounds: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    env_index = wp.tid()
    out_of_bounds[env_index] = (wp.abs(joint_pos[env_index, cart_dof_idx]) > max_cart_pos) or (
        wp.abs(joint_pos[env_index, pole_dof_idx]) > math.pi / 2.0
    )
    time_out[env_index] = episode_length_buf[env_index] >= (max_episode_length - 1)
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
    reward: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    reward[env_index] = (
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
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    initial_pose_angle_range: wp.vec2f,
    env_mask: wp.array(dtype=wp.bool),
    state: wp.array(dtype=wp.uint32),
):
    env_index = wp.tid()
    if env_mask[env_index]:
        joint_pos[env_index, cart_dof_idx] = default_joint_pos[env_index, cart_dof_idx]
        joint_pos[env_index, pole_dof_idx] = default_joint_pos[env_index, pole_dof_idx] + wp.randf(
            state[env_index], initial_pose_angle_range[0] * wp.pi, initial_pose_angle_range[1] * wp.pi
        )
        joint_vel[env_index, 0] = default_joint_vel[env_index, 0]
        joint_vel[env_index, 1] = default_joint_vel[env_index, 1]
        state[env_index] += wp.uint32(1)


@wp.kernel
def initialize_state(
    state: wp.array(dtype=wp.uint32),
    seed: wp.int32,
):
    env_index = wp.tid()
    state[env_index] = wp.rand_init(seed, env_index)


class CartpoleWarpEnv(DirectRLEnvWarp):
    cfg: CartpoleWarpEnvCfg

    def __init__(self, cfg: CartpoleWarpEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # Get the indices
        self._cart_dof_mask, _, self._cart_dof_idx = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_mask, _, self._pole_dof_idx = self.cartpole.find_joints(self.cfg.pole_dof_name)

        self.action_scale = self.cfg.action_scale

        # Simulation bindings
        # Note: these are direct memory views into the Newton simulation data, they should not be modified directly
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # Buffers
        self.observations = wp.zeros((self.num_envs), dtype=wp.vec4f, device=self.device)
        self.actions = wp.zeros((self.num_envs, 1), dtype=wp.float32, device=self.device)
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
        self.torch_episode_length_buf = wp.to_torch(self.episode_length_buf)

    def _setup_scene(self) -> None:
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: wp.array) -> None:
        wp.launch(
            update_actions,
            dim=self.num_envs,
            inputs=[
                actions,
                self.actions,
                self.action_scale,
            ],
        )

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_mask=self._cart_dof_mask)

    def _get_observations(self) -> None:
        wp.launch(
            get_observations,
            dim=self.num_envs,
            inputs=[
                self.joint_pos,
                self.joint_vel,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.observations,
            ],
        )

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
                self.rewards,
            ],
        )

    def _get_dones(self) -> None:
        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.joint_pos,
                self.episode_length_buf,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.max_episode_length,
                self.cfg.max_cart_pos,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
        )

    def _reset_idx(self, mask: wp.array | None = None) -> None:
        if mask is None:
            mask = self.cartpole._ALL_ENV_MASK

        super()._reset_idx(mask)

        wp.launch(
            reset,
            dim=self.num_envs,
            inputs=[
                self.cartpole.data.default_joint_pos,
                self.cartpole.data.default_joint_vel,
                self.joint_pos,
                self.joint_vel,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.cfg.initial_pole_angle_range,
                mask,
                self.states,
            ],
        )
