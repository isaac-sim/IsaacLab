# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-safe relative joint actions for conveyor transfer."""

from __future__ import annotations

import math
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
        if not math.isfinite(cfg.max_delta) or cfg.max_delta <= 0.0:
            raise ValueError("max_delta must be finite and positive.")
        if not math.isfinite(cfg.joint_limit_margin) or cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be finite and non-negative.")
        self._workspace_lower = torch.tensor(cfg.workspace_lower, dtype=torch.float32, device=self.device)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, dtype=torch.float32, device=self.device)
        if self._workspace_lower.shape != (self.action_dim,) or self._workspace_upper.shape != (self.action_dim,):
            raise ValueError("workspace bounds must contain one value per controlled joint.")
        if torch.any(self._workspace_lower >= self._workspace_upper):
            raise ValueError("Every lower workspace bound must be less than its upper bound.")
        if not torch.all(torch.isfinite(self._workspace_lower)) or not torch.all(torch.isfinite(self._workspace_upper)):
            raise ValueError("workspace bounds must be finite.")
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._position_targets = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()

    @property
    def previous_actions(self) -> torch.Tensor:
        """Previous finite policy actions, shape ``(num_envs, action_dim)``."""
        return self._previous_actions

    @property
    def invalid_actions(self) -> torch.Tensor:
        """Whether the latest policy action contained a non-finite component."""
        return self._invalid_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert normalized residuals into bounded position targets [rad]."""
        self._previous_actions.copy_(self._raw_actions)
        self._invalid_actions.copy_(~torch.isfinite(actions).all(dim=1))
        finite_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        super().process_actions(finite_actions)
        delta = torch.clamp(self._processed_actions, min=-self.cfg.max_delta, max=self.cfg.max_delta)
        positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = torch.maximum(limits[..., 0] + self.cfg.joint_limit_margin, self._workspace_lower)
        upper = torch.minimum(limits[..., 1] - self.cfg.joint_limit_margin, self._workspace_upper)
        self._position_targets = torch.clamp(positions + delta, min=lower, max=upper)
        self._processed_actions = self._position_targets

    def apply_actions(self) -> None:
        """Hold the policy-step target through all physics substeps."""
        self._asset.set_joint_position_target_index(target=self._position_targets, joint_ids=self._joint_ids)

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
        self._previous_actions[env_ids] = 0.0
        self._invalid_actions[env_ids] = False


class ResetBufferedGripperAction(BinaryJointPositionAction):
    """Keep reset-authored grasps closed during a short settling window."""

    cfg: ResetBufferedGripperActionCfg

    def __init__(self, cfg: ResetBufferedGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def previous_actions(self) -> torch.Tensor:
        """Previous finite policy actions, shape ``(num_envs, action_dim)``."""
        return self._previous_actions

    @property
    def invalid_actions(self) -> torch.Tensor:
        """Whether the latest gripper command contained a non-finite value."""
        return self._invalid_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map finite binary commands and preserve initially held cubes."""
        self._previous_actions.copy_(self._raw_actions)
        if actions.dtype == torch.bool:
            self._invalid_actions.zero_()
            finite_actions = actions
        else:
            self._invalid_actions.copy_(~torch.isfinite(actions).all(dim=1))
            # A non-finite gripper command closes the fingers for the final safe
            # step before the invalid-action termination is evaluated.
            finite_actions = torch.nan_to_num(actions, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        super().process_actions(finite_actions)
        command = self._env.command_manager.get_term(self.cfg.command_name)
        force_close = (command.held_cube_ids >= 0) & (self._env.episode_length_buf < self.cfg.force_close_steps)
        self._processed_actions[force_close] = self._close_command

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear buffered invalid-command state for selected environments."""
        super().reset(env_ids)
        self._previous_actions[env_ids] = 0.0
        self._invalid_actions[env_ids] = False
