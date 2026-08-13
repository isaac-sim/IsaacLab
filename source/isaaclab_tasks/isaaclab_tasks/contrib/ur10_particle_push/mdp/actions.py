# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 joint-delta action applied once per policy step."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import ClampedRelativeJointPositionActionCfg


class ClampedRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply one finite, limit-bounded relative joint target per policy step.

    The stock relative action adds its delta when actions are applied. This term instead snapshots
    the measured joint position while processing the policy action, matching DirectRL semantics
    when one policy step contains multiple physics substeps.
    """

    cfg: ClampedRelativeJointPositionActionCfg

    def __init__(self, cfg: ClampedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_limit_margin = float(cfg.joint_limit_margin)
        if not math.isfinite(self._joint_limit_margin) or self._joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be finite and non-negative.")
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._joint_targets = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

    @property
    def previous_actions(self) -> torch.Tensor:
        """Previous finite policy actions, shape ``(num_envs, action_dim)``."""
        return self._previous_actions

    @property
    def invalid_actions(self) -> torch.Tensor:
        """Whether the latest policy action contained a non-finite component."""
        return self._invalid_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize the policy action and construct one bounded joint target."""
        self._previous_actions.copy_(self._raw_actions)
        self._invalid_actions.copy_(~torch.isfinite(actions).all(dim=1))
        self._raw_actions.copy_(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))
        self._processed_actions.copy_(self._raw_actions * self._scale + self._offset)

        current_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = limits[..., 0] + self._joint_limit_margin
        upper = limits[..., 1] - self._joint_limit_margin
        self._joint_targets.copy_(
            torch.maximum(torch.minimum(current_position + self._processed_actions, upper), lower)
        )

    def apply_actions(self) -> None:
        """Apply the target computed once for the current policy step."""
        self._asset.set_joint_position_target_index(target=self._joint_targets, joint_ids=self._joint_ids)

    def set_reset_position(
        self,
        position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Align selected held targets with physical reset positions [rad]."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._joint_targets[selected].shape
        if position.shape != expected_shape:
            raise ValueError(f"Reset-position shape {tuple(position.shape)} does not match {tuple(expected_shape)}.")
        self._joint_targets[selected] = position.to(device=self.device, dtype=self._joint_targets.dtype)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear selected action and validity history without changing reset targets."""
        selected = slice(None) if env_ids is None else env_ids
        super().reset(selected)
        self._processed_actions[selected] = 0.0
        self._previous_actions[selected] = 0.0
        self._invalid_actions[selected] = False
