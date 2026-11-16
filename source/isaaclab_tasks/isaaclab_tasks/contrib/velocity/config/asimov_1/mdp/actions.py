# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import JointZeroOffsetPositionActionCfg


class JointZeroOffsetPositionAction(JointPositionAction):
    """Joint position action whose zero is shifted by a fixed calibration offset for each joint.

    This models joint encoders with a zero-position error. Each environment samples one offset [rad] for each
    action joint at start-up and keeps it. The offset is added to the position targets, and
    :func:`~.observations.joint_pos_rel_measured` subtracts it from the measured positions, so a zero action
    targets the pose that the policy observes as the default pose.
    """

    cfg: JointZeroOffsetPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointZeroOffsetPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        zero_offset = torch.empty_like(self._raw_actions).uniform_(*cfg.zero_offset_range)
        self._joint_zero_offset = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        self._joint_zero_offset[:, self._joint_ids] = zero_offset
        self._offset = self._offset + zero_offset

    @property
    def joint_zero_offset(self) -> torch.Tensor:
        """Joint-zero offset [rad] of every articulation joint, zero for joints outside this action term.

        Shape is (num_envs, num_joints).
        """
        return self._joint_zero_offset
