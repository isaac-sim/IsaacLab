# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG_DIRECT

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationDirect, ArticulationDirectCfg
from isaaclab.envs import DirectRLEnvDirect, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
import warp as wp

from isaaclab.assets.articulation_direct.kernels.other_kernels import generate_mask_from_ids


@configclass
class CartpoleDirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    solver_cfg = MJWarpSolverCfg(
        njmax=5,
        ls_iterations=3,
        cone="pyramidal",
        impratio=1,
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, newton_cfg=newton_cfg)

    # robot
    robot_cfg: ArticulationDirectCfg = CARTPOLE_CFG_DIRECT.replace(prim_path="/World/envs/env_.*/Robot")
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
    joint_pos: wp.array(dtype=wp.float32),
    joint_vel: wp.array(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    observations: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    observations[env_index, 0] = joint_pos[env_index, pole_dof_idx]
    observations[env_index, 1] = joint_vel[env_index, pole_dof_idx]
    observations[env_index, 2] = joint_pos[env_index, cart_dof_idx]
    observations[env_index, 3] = joint_vel[env_index, cart_dof_idx]

@wp.kernel
def update_actions(
    actions: wp.array(dtype=wp.vec2f),
    action_scale: wp.float32,
    cart_dof_idx: wp.int32,
):
    env_index = wp.tid()
    actions[env_index, cart_dof_idx] = action_scale * actions[env_index, cart_dof_idx]

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
):
    env_index = wp.tid()
    out_of_bounds[env_index] = (wp.abs(joint_pos[env_index, cart_dof_idx]) > max_cart_pos) or (wp.abs(joint_pos[env_index, pole_dof_idx]) > math.pi / 2)
    time_out[env_index] = episode_length_buf[env_index] >= (max_episode_length - 1)

class CartpoleDirectEnv(DirectRLEnvDirect):
    def __init__(self, cfg: CartpoleDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get the indices
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        # Convert to mask
        self._cart_dof_mask = wp.zeros(self.cartpole.num_joints, dtype=wp.bool, device=self.device)
        self._pole_dof_mask = wp.zeros(self.cartpole.num_joints, dtype=wp.bool, device=self.device)

        wp.launch(
            generate_mask_from_ids,
            dim=len(self._cart_dof_idx),
            inputs=[
                self._cart_dof_mask,
                wp.array(self._cart_dof_idx, dtype=wp.int32, device=self.device),
            ]
        )
        wp.launch(
            generate_mask_from_ids,
            dim=len(self._pole_dof_idx),
            inputs=[
                self._pole_dof_mask,
                wp.array(self._pole_dof_idx, dtype=wp.int32, device=self.device),
            ]
        )

        self.action_scale = self.cfg.action_scale
        # Get the joint position and velocity: Note this is a view into the articulation data, so no need to copy
        # This values should not be modified directly, but rather through the articulation class or kernels.
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # Buffers
        self.observations = wp.zeros((self.num_envs), dtype=wp.vec4f, device=self.device)
        self.actions = wp.zeros((self.num_envs), dtype=wp.vec2f, device=self.device)
        self.out_of_bounds = wp.zeros((self.num_envs), dtype=wp.bool, device=self.device)
        self.time_out = wp.zeros((self.num_envs), dtype=wp.bool, device=self.device)
        self.env_mask = wp.zeros((self.num_envs), dtype=wp.bool, device=self.device)
        self.rewards = wp.zeros((self.num_envs), dtype=wp.float32, device=self.device)

    def _setup_scene(self):
        self.cartpole = ArticulationDirect(self.cfg.robot_cfg)
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        wp.launch(
            update_actions,
            dim=self.num_envs,
            inputs=[
                self.actions,
                self.action_scale,
                self._cart_dof_idx[0],
            ]
        )

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_mask=self._cart_dof_mask)

    def _get_observations(self) -> dict:
        wp.launch(
            get_observations,
            dim=self.num_envs,
            inputs=[
                self.joint_pos,
                self.joint_vel,
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.observations,
            ]
        )
        observations = {"policy": wp.to_torch(self.observations)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
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
                wp.from_torch(self.reset_terminated),
                self.rewards,
            ]
        )
        return wp.to_torch(self.rewards)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.joint_pos,
                wp.from_torch(self.episode_length_buf),
                self._cart_dof_idx[0],
                self._pole_dof_idx[0],
                self.max_episode_length,
                self.cfg.max_cart_pos,
                self.out_of_bounds,
                self.time_out,
            ]
        )
        return wp.to_torch(self.out_of_bounds), wp.to_torch(self.time_out)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_mask = self.cartpole._ALL_ENV_MASK
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, wp.array):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
            wp.launch(
                generate_mask_from_ids,
                dim=(len(env_ids),),
                inputs=[
                    self.env_mask,
                    env_ids,
                ]
            )
        super()._reset_idx(env_ids)

        joint_pos = wp.to_torch(self.cartpole.data.default_joint_pos)[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = wp.to_torch(self.cartpole.data.default_joint_vel)[env_ids]

        default_root_state = wp.to_torch(self.cartpole.data.default_root_state)[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@wp.func
def compute_rew_alive(
    rew_scale_alive: wp.float32,
    reset_terminated: bool
) -> wp.float32:
    if reset_terminated:
        return wp.float32(0.0)
    return rew_scale_alive

@wp.func
def compute_rew_termination(
    rew_scale_terminated: wp.float32,
    reset_terminated: bool
) -> wp.float32:
    if reset_terminated:
        return rew_scale_terminated
    return wp.float32(0.0)

@wp.func
def compute_rew_pole_pos(
    rew_scale_pole_pos: wp.float32,
    pole_pos: wp.array(dtype=wp.float32),
) -> wp.float32:
    return rew_scale_pole_pos * pole_pos * pole_pos

@wp.func
def compute_rew_cart_vel(
    rew_scale_cart_vel: wp.float32,
    cart_vel: wp.array(dtype=wp.float32),
) -> wp.float32:
    return rew_scale_cart_vel * wp.abs(cart_vel)

@wp.func
def compute_rew_pole_vel(
    rew_scale_pole_vel: wp.float32,
    pole_vel: wp.array(dtype=wp.float32),
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
    reward[env_index] = compute_rew_alive(rew_scale_alive, reset_terminated) + \
        compute_rew_termination(rew_scale_terminated, reset_terminated) + \
        compute_rew_pole_pos(rew_scale_pole_pos, joint_pos[env_index, pole_dof_idx]) + \
        compute_rew_cart_vel(rew_scale_cart_vel, joint_vel[env_index, cart_dof_idx]) + \
        compute_rew_pole_vel(rew_scale_pole_vel, joint_vel[env_index, pole_dof_idx])

