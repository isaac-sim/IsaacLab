# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.controllers.utils import load_torchscript_model
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .configs.action_cfg import JointPositionPolicyActionCfg


class JointPositionPolicyAction(JointPositionAction):
    """Joint action term that applies the processed actions from a locomotion policy to the articulation's joints as position commands."""

    cfg: JointPositionPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointPositionPolicyActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # Load policy here if needed
        self._policy = load_torchscript_model(cfg.policy_path, device=env.device)
        self._env = env

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions using the locomotion policy.

        Args:
            actions: The input actions tensor.
        """
        # Get the cached observation from the environment
        obs_dict = self._env.obs_buf
        # Generate new actions from policy
        actions = self._policy(obs_dict["policy"])
        self._raw_actions[:] = actions

        # Apply scaling and offset to the raw actions from the policy
        self._processed_actions = self._raw_actions * self._scale + self._offset

        # Clip actions if configured
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
