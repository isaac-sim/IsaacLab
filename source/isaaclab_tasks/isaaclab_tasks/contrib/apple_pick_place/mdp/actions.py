# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GravityCompensationAction(ActionTerm):
    """Hold joints against gravity with feedforward efforts, taking no policy input.

    The policy does not command these joints. Each step, their effort targets are set to the articulation's
    gravity compensation forces, which the actuator adds to its position-target PD effort.
    """

    cfg: GravityCompensationActionCfg

    def __init__(self, cfg: GravityCompensationActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        # The gravity vector lists the floating base's degrees of freedom before the joints.
        self._gravity_ids = [i + self._asset.num_base_dofs for i in self._joint_ids]
        self._no_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._no_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._no_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        gravity = self._asset.data.gravity_compensation_forces.torch[:, self._gravity_ids]
        self._asset.actuators.target_command.set_effort_index(value=gravity, joint_ids=self._joint_ids)


@configclass
class GravityCompensationActionCfg(ActionTermCfg):
    """Configuration for :class:`GravityCompensationAction`."""

    class_type: type[ActionTerm] = GravityCompensationAction

    joint_names: list[str] = MISSING
    """Names or regular expressions of the joints to hold."""
