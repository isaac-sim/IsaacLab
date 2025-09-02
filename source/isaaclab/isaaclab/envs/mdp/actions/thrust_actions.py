# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from .joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg


# class JointEffortAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

#     cfg: actions_cfg.JointEffortActionCfg
#     """The configuration of the action term."""

#     def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#     def apply_actions(self):
#         # set joint effort targets
#         self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)


class JointThrustAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as thrust commands."""

    cfg: actions_cfg.JointThrustActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointThrustActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint thrust targets
        self._asset.set_thrust_target(self.processed_actions, joint_ids=self._joint_ids)

