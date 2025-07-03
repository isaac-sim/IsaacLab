# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AssemblyKitEnv: direct‐RL environment for the Franka assembly‐kit benchmark."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .peg_insertion_side_env_cfg import PegInsertionSideEnvCfg


class PegInsertionSideEnv(DirectRLEnv):

    cfg: PegInsertionSideEnvCfg

    def __init__(
        self, cfg: PegInsertionSideEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # random.seed(self.cfg.seed)

        self.joint_ids, _ = self.robot.find_joints("panda_joint.*|panda_finger_joint.*")

        # Find relevant link indices for runtime TCP computation
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]

    def _load_default_scene(self):

        # Creating the default scene
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0)
        )

        self.robot = Articulation(self.cfg.robot_cfg)

        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_scene(self):

        self._load_default_scene()

        # Filtering collisions for optimization of collisions between environment instances
        self.scene.filter_collisions(
            [
                "/World/ground",
            ]
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    # TODO choose the robot actuation method
    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions, joint_ids=self.joint_ids)
        # self.robot.set_joint_position_target(self.actions, joint_ids=self.joint_ids)

    # TODO implement the observation function
    def _get_observations(self) -> dict:

        state_obs = torch.cat(
            (torch.zeros((self.num_envs, 1), device=self.device), self.actions),
            dim=-1,
        )

        rgb = self.scene.sensors["camera"].data.output["rgb"]
        pixels = (rgb.to(torch.float32) / 255.0).clone()  # normalize to [0,1]

        # return according to obs_mode
        if self.cfg.obs_mode == "state":
            return {"policy": state_obs}
        else:
            return {"policy": pixels}

    # TODO implement the reward function
    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs,), device=self.device)
        return total_reward

    # TODO: Implement the termination and timeout logic
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.is_success()

        return terminated, time_out

    # TODO: Implement the success function
    def is_success(self) -> torch.Tensor:

        return torch.zeros((self.num_envs,), device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Resetting the robot
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
