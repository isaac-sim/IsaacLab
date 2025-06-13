
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

from isaaclab.envs.mdp.actions import JointAction, ActionTerm
from . import dextrah_kuka_allegro_constants as constants

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .action_cfg import LimitsScaledJointPositionActionCfg, PCAHandActionCfg


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


def compute_absolute_action(
    raw_actions: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    # Apply actions to hand
    absolute_action = scale(
        x=raw_actions,
        lower=lower_limits,
        upper=upper_limits,
    )
    absolute_action = tensor_clamp(
        t=absolute_action,
        min_t=lower_limits,
        max_t=upper_limits,
    )

    return absolute_action

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


class PCAAction(ActionTerm):
    
    cfg: PCAHandActionCfg
    
    def __init__(self, cfg:PCAHandActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        cfg.arm_joints_cfg.resolve(env.scene)
        cfg.pca_joints_cfg.resolve(env.scene)
        self.num_arm_joints = len(cfg.arm_joints_cfg.joint_ids)
        self.num_hand_joints = len(cfg.pca_joints_cfg.joint_ids)
        self.pca_matrix = torch.tensor(cfg.pca_matrix, device=self.device)
        self.arm_joint_actions = torch.zeros((self.num_envs, self.num_arm_joints), device=self.device)
        self._raw_actions = torch.zeros((self.num_envs, self.pca_matrix.shape[0] + self.num_arm_joints), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, self.num_arm_joints + self.num_hand_joints), device=self.device)
        self.hand_pca_upper_limits = to_torch(constants.HAND_PCA_MAXS, device=self.device)
        self.hand_pca_lower_limits = to_torch(constants.HAND_PCA_MINS, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.pca_matrix.shape[0] + self.num_arm_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # relative
        self._processed_actions[:, self.cfg.arm_joints_cfg.joint_ids] = actions[:, :self.num_arm_joints] * 0.3 + self._asset.data.joint_pos[:, self.cfg.arm_joints_cfg.joint_ids]
        hand_pca_target = compute_absolute_action(self._raw_actions[:, self.num_arm_joints:], self.hand_pca_lower_limits, self.hand_pca_upper_limits)
        self._processed_actions[:, self.cfg.pca_joints_cfg.joint_ids] = hand_pca_target @ self.pca_matrix
    
    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions)
