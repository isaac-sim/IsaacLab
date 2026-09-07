# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware joint action terms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm

from ..selection_utils import SceneEntitySelectionCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import SelectedBinaryJointPositionActionCfg, SelectedJointPositionActionCfg


class _SelectedJointAction(ActionTerm):
    """Resolve joints and maintain global inputs for a partial articulation view."""

    cfg: SelectedBinaryJointPositionActionCfg | SelectedJointPositionActionCfg
    _asset: Articulation

    def __init__(
        self, cfg: SelectedBinaryJointPositionActionCfg | SelectedJointPositionActionCfg, env: ManagerBasedEnv
    ) -> None:
        super().__init__(cfg, env)
        joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        self._num_joints = len(joint_ids)
        self._asset_cfg = SceneEntitySelectionCfg(cfg.asset_name)
        self._asset_cfg.resolve(env.scene)
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Unprocessed global action batch."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions in asset physics-view order."""
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear raw actions for the reset global environments."""
        self._raw_actions[env_ids] = 0.0


class SelectedJointPositionAction(_SelectedJointAction):
    """Apply a global action batch to one articulation's selected physics-view rows."""

    cfg: SelectedJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: SelectedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_limit_margin = cfg.joint_limit_margin
        if self._joint_limit_margin is not None:
            if not math.isfinite(self._joint_limit_margin) or self._joint_limit_margin < 0.0:
                raise ValueError("joint_limit_margin must be finite and non-negative.")
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            if torch.any(limits[..., 0] + self._joint_limit_margin > limits[..., 1] - self._joint_limit_margin):
                raise ValueError("joint_limit_margin leaves an empty target range for at least one joint.")
        self._processed_actions = torch.zeros(len(self._asset_cfg.env_ids), self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        """Number of controlled joints."""
        return self._num_joints

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale global actions and align them with the articulation physics view."""
        self._raw_actions[:] = actions
        self._processed_actions = actions[self._asset_cfg.env_ids] * self.cfg.scale
        if not self.cfg.relative:
            self._processed_actions += self._asset.data.default_joint_pos.torch[:, self._joint_ids]

    def apply_actions(self) -> None:
        """Write joint-position targets to the selected articulation instances."""
        target = self._processed_actions
        if self.cfg.relative:
            target = target + self._asset.data.joint_pos.torch[:, self._joint_ids]
        if self._joint_limit_margin is not None:
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            target = torch.maximum(
                torch.minimum(target, limits[..., 1] - self._joint_limit_margin),
                limits[..., 0] + self._joint_limit_margin,
            )
        self._asset.actuators.target_command.set_position_index(value=target, joint_ids=self._joint_ids)


class SelectedBinaryJointPositionAction(_SelectedJointAction):
    """Map one binary global action to joint targets in asset physics-view order."""

    cfg: SelectedBinaryJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: SelectedBinaryJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._processed_actions = torch.zeros(len(self._asset_cfg.env_ids), self._num_joints, device=self.device)
        self._open_command = torch.full((self._num_joints,), cfg.open_command, device=self.device)
        self._close_command = torch.full((self._num_joints,), cfg.close_command, device=self.device)

    @property
    def action_dim(self) -> int:
        """Binary action dimension."""
        return 1

    def process_actions(self, actions: torch.Tensor) -> None:
        """Choose open or closed targets from the sign of selected actions."""
        self._raw_actions[:] = actions
        self._processed_actions = torch.where(
            actions[self._asset_cfg.env_ids] < 0.0, self._close_command, self._open_command
        )

    def apply_actions(self) -> None:
        """Write selected binary joint targets."""
        self._asset.actuators.target_command.set_position_index(
            value=self._processed_actions, joint_ids=self._joint_ids
        )
