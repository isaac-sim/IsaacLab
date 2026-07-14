# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg

from isaaclab.actuators import ImplicitActuator

from isaaclab_tasks.core.locomotion.ant.ant_direct_env_cfg import AntEnvCfg
from isaaclab_tasks.core.locomotion.ant.ant_post_step import _AntPostStepBuffers
from isaaclab_tasks.core.locomotion.locomotion_direct_env import LocomotionDirectEnv


class AntEnv(LocomotionDirectEnv):
    """Direct-workflow Ant locomotion environment."""

    cfg: AntEnvCfg

    def __init__(self, cfg: AntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._use_fused_post_step = self._supports_fused_post_step()
        if not self._use_fused_post_step:
            return

        # Cache zero-copy state views once. The backing simulation arrays remain stable
        # for the lifetime of the environment.
        self.torso_position = self.robot.data.root_pos_w.torch
        self.torso_rotation = self.robot.data.root_quat_w.torch
        self.velocity = self.robot.data.root_lin_vel_w.torch
        self.ang_velocity = self.robot.data.root_ang_vel_w.torch
        self.dof_pos = self.robot.data.joint_pos.torch
        self.dof_vel = self.robot.data.joint_vel.torch
        self._joint_position_limits = self.robot.data.soft_joint_pos_limits.warp

        self._post_step_buffers = _AntPostStepBuffers(
            num_envs=self.num_envs,
            num_joints=self.robot.num_joints,
            observation_size=self.cfg.observation_space,
            device=self.device,
        )
        self._post_step_buffers.bind_environment_outputs(self)
        self._default_root_pose_w_torch = self.robot.data.default_root_pose.torch.clone()
        self._default_root_pose_w_torch[:, :3] += self.scene.env_origins
        self._default_root_pose_w = wp.from_torch(self._default_root_pose_w_torch, dtype=wp.transformf)

    def _compute_intermediate_values(self) -> None:
        if not self._use_fused_post_step:
            return super()._compute_intermediate_values()
        self._post_step_buffers.compute_intermediate_and_observation(self)
        return None

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if not self._use_fused_post_step:
            return super()._get_observations()
        return {"policy": self._post_step_buffers.observation_torch}

    def _get_rewards(self) -> torch.Tensor:
        if not self._use_fused_post_step:
            return super()._get_rewards()
        return self._post_step_buffers.reward_torch

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._use_fused_post_step:
            return super()._get_dones()
        self._post_step_buffers.compute_post_step(self)
        return self._post_step_buffers.terminated_torch, self._post_step_buffers.time_out_torch

    def _reset_envs_from_buffer(self) -> torch.Tensor | None:
        if not self._supports_mask_reset():
            return super()._reset_envs_from_buffer()

        env_mask = wp.from_torch(self.reset_buf)
        self._reset_idx_mask(env_mask)
        return None

    def _supports_mask_reset(self) -> bool:
        """Return whether the current Ant configuration supports synchronization-free mask resets."""
        return (
            self._use_fused_post_step
            and not getattr(self.robot, "_has_newton_actuators", False)
            and all(isinstance(actuator, ImplicitActuator) for actuator in self.robot.actuators.values())
            and not self.cfg.events
            and not self.cfg.action_noise_model
            and not self.cfg.observation_noise_model
            and not (self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0)
        )

    def _supports_fused_post_step(self) -> bool:
        """Return whether state views and terminal outputs can safely use the fused Newton path."""
        return isinstance(self.cfg.sim.physics, NewtonCfg) and not self.cfg.compute_final_obs

    def _reset_idx_mask(self, env_mask: wp.array(dtype=wp.bool)) -> None:
        """Reset Newton Ant environments selected by a boolean device mask."""
        reset_count = self.reset_buf.sum()
        survived_count = torch.logical_and(self.reset_time_outs, self.reset_buf).sum()
        success_rate = survived_count.float() / reset_count.clamp_min(1)
        log = self.extras.setdefault("log", {})
        previous_success_rate = log.get("Metrics/success_rate", torch.zeros((), device=self.device))
        if not isinstance(previous_success_rate, torch.Tensor):
            previous_success_rate = torch.tensor(previous_success_rate, device=self.device)
        log["Metrics/success_rate"] = torch.where(reset_count > 0, success_rate, previous_success_rate)

        # Ant uses stateless implicit actuators, so only its wrench buffers need reset.
        if self.robot.instantaneous_wrench_composer.active:
            self.robot.instantaneous_wrench_composer.reset(env_mask=env_mask)
        if self.robot.permanent_wrench_composer.active:
            self.robot.permanent_wrench_composer.reset(env_mask=env_mask)
        self.robot.write_root_pose_to_sim_mask(root_pose=self._default_root_pose_w, env_mask=env_mask)
        self.robot.write_root_velocity_to_sim_mask(
            root_velocity=self.robot.data.default_root_vel.warp,
            env_mask=env_mask,
        )
        self.robot.write_joint_state_to_sim_mask(
            position=self.robot.data.default_joint_pos.warp,
            velocity=self.robot.data.default_joint_vel.warp,
            env_mask=env_mask,
        )
        self._post_step_buffers.compute_masked_reset_observation(self, env_mask)
