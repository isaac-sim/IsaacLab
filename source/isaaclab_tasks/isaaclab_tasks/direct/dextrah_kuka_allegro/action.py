
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import LimitsScaledJointPositionActionCfg


class LimitsScaledJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: LimitsScaledJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: LimitsScaledJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        self.low_lim = self._asset.data.joint_pos_limits[:, :, 0]
        self.high_lim = self._asset.data.joint_pos_limits[:, :, 1]
        self._prev_action = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._prev_action = self._processed_actions.clone()
        proc_actions = self._raw_actions.clamp(min=-1.0, max=1.0)
        proc_actions = torch.lerp(self.low_lim, self.high_lim, (proc_actions  + 1) * 0.5)
        self._processed_actions = self.cfg.ema_lambda * proc_actions + (1 - self.cfg.ema_lambda) * self._prev_action
    
    def reset(self, env_ids = None):
        self._prev_action[env_ids] = 0.0

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)