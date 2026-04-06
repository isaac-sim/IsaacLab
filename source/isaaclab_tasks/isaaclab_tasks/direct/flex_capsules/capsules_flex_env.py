# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import warp as wp
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation #, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .capsules_flex_env_cfg import CapsulesFlexEnvCfg


class CapsulesFlexEnv(DirectRLEnv):
    cfg: CapsulesFlexEnvCfg

    def __init__(self, cfg: CapsulesFlexEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.write_actuator_ctrl_to_sim_index(ctrl=wp.array([-1.0, -1.0], dtype=wp.float32))

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, 0].unsqueeze(dim=1),
                self.joint_pos[:, 1].unsqueeze(dim=1),
                self.joint_vel[:, 0].unsqueeze(dim=1),
                self.joint_vel[:, 1].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, 0],
            self.joint_vel[:, 0],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple(torch.Tensor,torch.Tensor):
        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos) > math.pi , dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids]
        self.joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids]

        default_root_state = wp.to_torch(self.robot.data.default_root_state)[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_actuator_ctrl_to_sim_index(ctrl=wp.to_torch(self.robot.data._default_actuator_ctrl))

        # self.robot.write_root_pose_to_sim_index(default_root_state[:, :7], env_ids)
        # self.robot.write_root_velocity_to_sim_index(default_root_state[:, 7:], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # self.robot.write_fixed_tendon_properties_to_sim_index()


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_pole_vel
    return total_reward