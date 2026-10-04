# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BiasedJointPositionAction(JointPositionAction):
    """Joint position action that compensates the joint-encoder calibration error."""

    cfg: BiasedJointPositionActionCfg
    """The configuration of the action term."""

    encoder_bias: torch.Tensor
    """Encoder calibration error per environment and action joint [rad], shape (num_envs, action_dim)."""

    def __init__(self, cfg: BiasedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.encoder_bias = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def apply_actions(self):
        target = self.processed_actions - self.encoder_bias
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the encoder-bias-compensating joint position action term."""

    class_type: type[BiasedJointPositionAction] = BiasedJointPositionAction
