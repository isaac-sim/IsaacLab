# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerically safe joint actions for cable routing."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction, RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass


def _finite_unit_actions(actions: torch.Tensor) -> torch.Tensor:
    """Replace non-finite policy outputs and enforce the policy's unit action range."""
    return torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


class FiniteRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that cannot forward non-finite targets to physics."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(_finite_unit_actions(actions))


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
