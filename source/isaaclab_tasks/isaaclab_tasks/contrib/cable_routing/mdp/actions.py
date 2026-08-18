# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerically safe joint actions for cable routing."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction, RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass


def _finite_unit_actions(actions: torch.Tensor) -> torch.Tensor:
    """Replace non-finite policy outputs and enforce the policy's unit action range."""
    return torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def canonical_task_actions(env, actions: torch.Tensor, binary_action_names: Sequence[str]) -> torch.Tensor:
    """Return finite unit actions with binary terms represented only by their selected state."""
    actions = _finite_unit_actions(actions)
    binary_names = set(binary_action_names)
    unknown_names = binary_names.difference(env.action_manager.active_terms)
    if unknown_names:
        raise ValueError(f"Unknown binary action terms: {sorted(unknown_names)}.")

    chunks: list[torch.Tensor] = []
    offset = 0
    for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        term_actions = actions[:, offset : offset + dim]
        if name in binary_names:
            term_actions = torch.where(
                term_actions < 0.0, -torch.ones_like(term_actions), torch.ones_like(term_actions)
            )
        chunks.append(term_actions)
        offset += dim
    return torch.cat(chunks, dim=1)


class FiniteRelativeJointPositionAction(RelativeJointPositionAction):
    """Finite relative joint action that holds one limit-clamped target per policy step."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(_finite_unit_actions(actions))

        # The stock relative action adds the delta to the live joint position in apply_actions(),
        # which is called once per simulation step. Resolve the absolute target here instead so
        # control semantics do not change with environment decimation or Newton graph capture.
        current = self._asset.data.joint_pos.torch[:, self._joint_ids]
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        current = torch.where(torch.isfinite(current), current, default)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = current + self._processed_actions
        target = torch.where(torch.isfinite(target), target, default)
        self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)


class FiniteBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper action that cannot forward non-finite commands to physics."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(_finite_unit_actions(actions))


@configclass
class FiniteRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`FiniteRelativeJointPositionAction`."""

    class_type: type[FiniteRelativeJointPositionAction] = FiniteRelativeJointPositionAction


@configclass
class FiniteBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for :class:`FiniteBinaryJointPositionAction`."""

    class_type: type[FiniteBinaryJointPositionAction] = FiniteBinaryJointPositionAction
