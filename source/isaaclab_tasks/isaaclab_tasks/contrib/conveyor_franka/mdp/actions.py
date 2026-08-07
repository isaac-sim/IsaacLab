# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-safe relative joint actions for conveyor transfer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import ConveyorRelativeJointPositionActionCfg, ResetBufferedGripperActionCfg


class ConveyorRelativeJointPositionAction(JointAction):
    """Apply one measured-state-relative target per policy step.

    Isaac Lab's generic relative action adds the residual during every physics
    substep. This term computes the target once in :meth:`process_actions`, so
    simulation decimation does not multiply the requested displacement.
    """

    cfg: ConveyorRelativeJointPositionActionCfg

    def __init__(self, cfg: ConveyorRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.max_delta <= 0.0:
            raise ValueError("max_delta must be positive.")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative.")
        self._workspace_lower = torch.tensor(cfg.workspace_lower, dtype=torch.float32, device=self.device)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, dtype=torch.float32, device=self.device)
        if self._workspace_lower.shape != (self.action_dim,) or self._workspace_upper.shape != (self.action_dim,):
            raise ValueError("workspace bounds must contain one value per controlled joint.")
        if torch.any(self._workspace_lower >= self._workspace_upper):
            raise ValueError("Every lower workspace bound must be less than its upper bound.")
        resolved_joint_ids = (
            list(range(self._asset.num_joints)) if isinstance(self._joint_ids, slice) else self._joint_ids
        )
        self._gravity_joint_ids = [joint_id + self._asset.num_base_dofs for joint_id in resolved_joint_ids]
        self._position_targets = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert normalized residuals into bounded position targets [rad]."""
        super().process_actions(actions)
        delta = torch.clamp(self._processed_actions, min=-self.cfg.max_delta, max=self.cfg.max_delta)
        positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = torch.maximum(limits[..., 0] + self.cfg.joint_limit_margin, self._workspace_lower)
        upper = torch.minimum(limits[..., 1] - self.cfg.joint_limit_margin, self._workspace_upper)
        self._position_targets = torch.clamp(positions + delta, min=lower, max=upper)
        self._processed_actions = self._position_targets
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        """Hold the policy-step target and gravity feedforward through all physics substeps."""
        self._asset.set_joint_position_target_index(target=self._position_targets, joint_ids=self._joint_ids)
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._gravity_joint_ids]
            gravity = torch.where(torch.isfinite(gravity), gravity, torch.zeros_like(gravity))
            self._asset.set_joint_effort_target_index(target=gravity, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Initialize targets from the sampled reset pose."""
        super().reset(env_ids)
        positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if env_ids is None:
            self._position_targets[:] = positions
            self._processed_actions[:] = positions
        else:
            self._position_targets[env_ids] = positions[env_ids]
            self._processed_actions[env_ids] = positions[env_ids]


class ResetBufferedGripperAction(BinaryJointPositionAction):
    """Keep reset-authored grasps closed during a short settling window."""

    cfg: ResetBufferedGripperActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map binary commands and preserve initially held cubes."""
        super().process_actions(actions)
        state = getattr(self._env, "conveyor_transfer_state", None)
        if state is None:
            return
        force_close = (state.held_cube_ids >= 0) & (self._env.episode_length_buf < self.cfg.force_close_steps)
        self._processed_actions[force_close] = self._close_command
