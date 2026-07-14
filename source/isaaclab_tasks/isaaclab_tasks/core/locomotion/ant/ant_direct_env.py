# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.locomotion.ant.ant_direct_env_cfg import AntEnvCfg
from isaaclab_tasks.core.locomotion.ant.ant_post_step import _AntPostStepBuffers
from isaaclab_tasks.core.locomotion.locomotion_direct_env import LocomotionDirectEnv


class AntEnv(LocomotionDirectEnv):
    """Direct-workflow Ant locomotion environment."""

    cfg: AntEnvCfg

    def __init__(self, cfg: AntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._use_fused_post_step = self._supports_fused_post_step()
        self._fused_inputs_require_refresh = isinstance(self.cfg.sim.physics, PhysxCfg)
        if not self._use_fused_post_step:
            return

        # Cache slices of Tier-1 packed state. Newton component properties may materialize
        # staging buffers for strided views and only refresh them when accessed.
        root_link_pose_w = self.robot.data.root_link_pose_w.torch
        root_com_vel_w = self.robot.data.root_com_vel_w.torch
        self.torso_position = root_link_pose_w[:, :3]
        self.torso_rotation = root_link_pose_w[:, 3:7]
        self.velocity = root_com_vel_w[:, :3]
        self.ang_velocity = root_com_vel_w[:, 3:6]
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

    def _compute_intermediate_values(self) -> None:
        if not self._use_fused_post_step:
            return super()._compute_intermediate_values()
        self._refresh_fused_post_step_inputs()
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
        self._refresh_fused_post_step_inputs()
        self._post_step_buffers.compute_post_step(self)
        return self._post_step_buffers.terminated_torch, self._post_step_buffers.time_out_torch

    def _supports_fused_post_step(self) -> bool:
        """Return whether state views and terminal outputs can safely use the fused post-step path."""
        return isinstance(self.cfg.sim.physics, (NewtonCfg, PhysxCfg)) and not self.cfg.compute_final_obs

    def _refresh_fused_post_step_inputs(self) -> None:
        """Pull current PhysX state into the stable buffers consumed by the fused kernel."""
        if not self._fused_inputs_require_refresh:
            return

        data = self.robot.data
        _ = data.root_link_pose_w, data.root_com_vel_w, data.joint_pos, data.joint_vel
