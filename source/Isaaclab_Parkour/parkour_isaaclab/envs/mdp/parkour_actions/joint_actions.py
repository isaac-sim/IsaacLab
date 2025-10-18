
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import DelayedJointPositionActionCfg

class DelayedJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: DelayedJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._action_history_buf = torch.zeros(self.num_envs, cfg.history_length, self._num_joints, device=self.device, dtype=torch.float)
        self._delay_update_global_steps = cfg.delay_update_global_steps
        self._action_delay_steps = cfg.action_delay_steps
        self._use_delay = cfg.use_delay
        self.env = env 

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        if self.env.common_step_counter % self._delay_update_global_steps == 0:
            if len(self._action_delay_steps) != 0:
                self.delay = torch.tensor(self._action_delay_steps.pop(0), device=self.device, dtype=torch.float)
        self._action_history_buf = torch.cat([self._action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        indices = -1 - self.delay 
        if self._use_delay:
            self._raw_actions[:] = self._action_history_buf[:, indices.long()]
        else:
            self._raw_actions[:] = actions
        # apply the affine transformations

        if self.cfg.clip is not None:
            self._raw_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._action_history_buf[env_ids, :, :] = 0.

    @property
    def action_history_buf(self):
        return self._action_history_buf
