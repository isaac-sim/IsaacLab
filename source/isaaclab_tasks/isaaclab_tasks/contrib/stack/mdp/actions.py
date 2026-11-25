# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific actions for reset-table cube stacking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction

from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        ResetBufferedGripperActionCfg,
        WorkspaceBoundedRelativeJointPositionActionCfg,
    )


class ResetBufferedGripperAction(BinaryJointPositionAction):
    """Keep reset-supplied grasps closed only while the reset state settles.

    Newton Cube Lift keeps the physical gripper closed throughout the
    reset-assisted grasp portion of its curriculum. Stacking can use that
    protection only during the reset grace period: continuing to mask an open
    policy command through transport would credit an action that drops the
    cube in an uninterrupted table-start episode. Raw actions remain visible
    to the termination manager, and after the grace period the policy owns the
    physical gripper completely.
    """

    cfg: ResetBufferedGripperActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map binary policy actions and protect reset-assisted acquisition."""
        super().process_actions(actions)
        if self.cfg.force_close_steps == 0:
            return
        held_cube_ids = get_stack_reset_runtime_state(self._env).held_cube_ids
        force_close = (held_cube_ids >= 0) & (self._env.episode_length_buf < self.cfg.force_close_steps)
        self._processed_actions[force_close] = self._close_command


class WorkspaceBoundedRelativeJointPositionAction(JointAction):
    """Map policy joint deltas to gravity-compensated, bounded position targets.

    Each policy output directly controls the corresponding arm joint. A scaled
    delta is added to the measured joint position exactly once
    per policy step, then the resulting target is held through every physics
    substep. This preserves standard relative-joint semantics without applying
    the same delta repeatedly during simulation decimation. Targets are also
    clamped to both the robot's soft limits and the reset-buffer workspace.
    Model-based gravity effort can be added alongside the impedance target.
    """

    cfg: WorkspaceBoundedRelativeJointPositionActionCfg

    def __init__(self, cfg: WorkspaceBoundedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        # Use int64 indices for PyTorch reads and keep the backend indices for commands.
        self._torch_joint_ids = self._joint_ids.long() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        self._processed_actions.copy_(self._asset.data.joint_pos.torch[:, self._torch_joint_ids])
        self._gravity_joint_ids = (
            [joint_id + self._asset.num_base_dofs for joint_id in range(self._asset.num_joints)]
            if isinstance(self._torch_joint_ids, slice)
            else self._torch_joint_ids + self._asset.num_base_dofs
        )
        self._workspace_lower = torch.tensor(cfg.workspace_lower, device=self.device, dtype=torch.float32)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, device=self.device, dtype=torch.float32)
        if self._workspace_lower.shape != (self.action_dim,) or self._workspace_upper.shape != (self.action_dim,):
            raise ValueError(
                "workspace_lower and workspace_upper must each contain one value "
                f"per controlled joint ({self.action_dim})."
            )
        if torch.any(self._workspace_lower >= self._workspace_upper):
            raise ValueError("Every workspace_lower value must be less than workspace_upper.")

    def process_actions(self, actions: torch.Tensor) -> None:
        """Create one bounded position target from the measured joint pose."""
        super().process_actions(actions)
        target_delta = torch.clamp(
            self._processed_actions,
            min=-self.cfg.max_delta,
            max=self.cfg.max_delta,
        )
        current_position = self._asset.data.joint_pos.torch[:, self._torch_joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._torch_joint_ids]
        lower = torch.maximum(
            limits[..., 0] + self.cfg.joint_limit_margin,
            self._workspace_lower,
        )
        upper = torch.minimum(
            limits[..., 1] - self.cfg.joint_limit_margin,
            self._workspace_upper,
        )
        self._processed_actions[:] = torch.clamp(current_position + target_delta, min=lower, max=upper)

    def apply_actions(self) -> None:
        """Apply the fixed target and gravity feedforward during every physics substep."""
        command = self._asset.actuators.target_command
        command.set_position_index(value=self.processed_actions, joint_ids=self._joint_ids)
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._gravity_joint_ids]
            gravity = torch.where(torch.isfinite(gravity), gravity, torch.zeros_like(gravity))
            command.set_effort_index(value=gravity, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Initialize the held target from each sampled reset pose."""
        super().reset(env_ids)
        current_position = self._asset.data.joint_pos.torch[:, self._torch_joint_ids]
        selection = slice(None) if env_ids is None else env_ids
        self._processed_actions[selection] = current_position[selection]
