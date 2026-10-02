# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .events import encoder_bias

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BiasedJointPositionAction(JointPositionAction):
    """Joint position action that compensates the joint-encoder calibration error."""

    cfg: BiasedJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: BiasedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._encoder_bias = encoder_bias(env, SceneEntityCfg(cfg.asset_name))

    def apply_actions(self):
        target = self.processed_actions - self._encoder_bias[:, self._joint_ids]
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the encoder-bias-compensating joint position action term."""

    class_type: type[BiasedJointPositionAction] = BiasedJointPositionAction
