from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import sample_uniform

from .cartpole_camera_env_cfg import CartpoleCameraBaseEnvCfg


class CartpoleCameraEnv(DirectRLEnv):
    cfg: CartpoleCameraBaseEnvCfg

    def __init__(self, cfg: CartpoleCameraBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation and sensors to scene
        self.scene.articulations["cartpole"] = self.cartpole
        self.scene.sensors["tiled_camera"] = self.tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        # fundamental spaces
        # - Box
        if isinstance(self.single_action_space, gym.spaces.Box):
            target = self.cfg.max_effort * self.actions
        # - Discrete
        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            target = torch.where(self.actions == 1, -self.cfg.max_effort, target)
            target = torch.where(self.actions == 2, self.cfg.max_effort, target)
        # - MultiDiscrete
        elif isinstance(self.single_action_space, gym.spaces.MultiDiscrete):
            # value
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            target = torch.where(self.actions[:, [0]] == 1, self.cfg.max_effort / 2.0, target)
            target = torch.where(self.actions[:, [0]] == 2, self.cfg.max_effort, target)
            # direction
            target = torch.where(self.actions[:, [1]] == 0, -target, target)
        else:
            raise NotImplementedError(f"Action space {type(self.single_action_space)} not implemented")

        # set target
        self.cartpole.set_joint_effort_target(target, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        if "rgb" in self.cfg.tiled_camera.data_types:
            camera_data = self.tiled_camera.data.output[data_type] / 255.0
            # normalize the camera data for better training results
            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
        elif "depth" in self.cfg.tiled_camera.data_types:
            camera_data = self.tiled_camera.data.output[data_type]
            camera_data[camera_data == float("inf")] = 0

        # fundamental spaces
        # - Box
        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            obs = camera_data
        # composite spaces
        # - Tuple
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Tuple):
            obs = (self.joint_vel, camera_data)
        # - Dict
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            obs = {"joint-velocities": self.joint_vel, "camera": camera_data}

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
