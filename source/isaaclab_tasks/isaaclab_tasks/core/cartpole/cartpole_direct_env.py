# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, wrap_to_pi

from isaaclab_tasks.core.cartpole.cartpole_post_step import _CartpolePostStepBuffers

if TYPE_CHECKING:
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg


class CartpoleEnv(DirectRLEnv):
    """Cartpole balancing environment driven by proprioceptive (joint-state) observations."""

    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos.torch
        self.joint_vel = self.cartpole.data.joint_vel.torch
        self._use_fused_post_step = self._supports_fused_post_step()
        self._fused_inputs_require_refresh = isinstance(self.cfg.sim.physics, PhysxCfg)
        if self._use_fused_post_step:
            self._joint_position_warp = self.cartpole.data.joint_pos.warp
            self._joint_velocity_warp = self.cartpole.data.joint_vel.warp
            self._post_step_buffers = _CartpolePostStepBuffers(self.num_envs, self.device)

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
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
        # quaternion for euler angles (roll, pitch, yaw) = (0, -45, -45) degrees
        light_orientation = (-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
        light_cfg.func("/World/Light", light_cfg, orientation=light_orientation)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions
        self.cartpole.set_joint_effort_target_index(target=self.actions, joint_ids=self._cart_dof_idx)

    def _apply_action(self) -> None:
        # Joint effort targets persist in the articulation command buffer across decimation substeps.
        pass

    def _get_observations(self) -> dict:
        if self._use_fused_post_step:
            return {"policy": self._post_step_buffers.observation_torch}
        joint_pos_rel = self.joint_pos - self.cartpole.data.default_joint_pos.torch
        joint_vel_rel = self.joint_vel - self.cartpole.data.default_joint_vel.torch
        obs = torch.cat(
            (
                joint_pos_rel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                joint_pos_rel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                joint_vel_rel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                joint_vel_rel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        if self._use_fused_post_step:
            return self._post_step_buffers.reward_torch
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
            self.step_dt,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_fused_post_step:
            self._refresh_fused_post_step_inputs()
            self._post_step_buffers.compute_post_step(self)
            return self._post_step_buffers.terminated_torch, self._post_step_buffers.time_out_torch
        self.joint_pos = self.cartpole.data.joint_pos.torch
        self.joint_vel = self.cartpole.data.joint_vel.torch
        time_out = self.episode_length_buf >= self.max_episode_length
        out_of_bounds = torch.abs(self.joint_pos[:, self._cart_dof_idx[0]]) > self.cfg.max_cart_pos
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES

        # Log survival success rate before resetting
        survived = self.reset_time_outs[env_ids].float()
        self.extras.setdefault("log", {})["Metrics/success_rate"] = survived.mean().item()

        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._cart_dof_idx] += sample_uniform(
            self.cfg.initial_cart_position_range[0],
            self.cfg.initial_cart_position_range[1],
            joint_pos[:, self._cart_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel.torch[env_ids].clone()
        joint_vel[:, self._cart_dof_idx] += sample_uniform(
            self.cfg.initial_cart_velocity_range[0],
            self.cfg.initial_cart_velocity_range[1],
            joint_vel[:, self._cart_dof_idx].shape,
            joint_vel.device,
        )
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0],
            self.cfg.initial_pole_angle_range[1],
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_velocity_range[0],
            self.cfg.initial_pole_velocity_range[1],
            joint_vel[:, self._pole_dof_idx].shape,
            joint_vel.device,
        )

        # clamp the sampled state to the joint limits (matches the manager-based reset_joints_by_offset)
        joint_pos_limits = self.cartpole.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel_limits = self.cartpole.data.soft_joint_vel_limits.torch[env_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        default_root_pose = self.cartpole.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self.cartpole.data.default_root_vel.torch[env_ids].clone()

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.cartpole.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.cartpole.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.cartpole.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        if getattr(self, "_use_fused_post_step", False):
            self._post_step_buffers.compute_observation(self)

    def _supports_fused_post_step(self) -> bool:
        """Return whether live state can safely supply fused post-step calculations."""
        return (
            isinstance(self.cfg.sim.physics, (NewtonCfg, PhysxCfg))
            and self.cfg.observation_space == 4
            and not self.cfg.compute_final_obs
            and not self.cfg.events
            and type(self)._get_observations is CartpoleEnv._get_observations
            and type(self)._get_rewards is CartpoleEnv._get_rewards
            and type(self)._get_dones is CartpoleEnv._get_dones
            and type(self)._reset_idx is CartpoleEnv._reset_idx
        )

    def _refresh_fused_post_step_inputs(self) -> None:
        """Pull current PhysX joint state into the stable buffers consumed by the fused kernel."""
        if not self._fused_inputs_require_refresh:
            return

        data = self.cartpole.data
        _ = data.joint_pos, data.joint_vel


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
):
    pole_pos = wrap_to_pi(pole_pos)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = (rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel) * step_dt
    return total_reward
