# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from .joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

class ThrustAction(JointAction):
    """Joint action term that applies the processed actions as thrust commands."""

    cfg: actions_cfg.ThrustActionCfg
    """The configuration of the action term."""
    

    def __init__(self, cfg: actions_cfg.ThrustActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint thrust targets
        self._asset.set_thrust_target(self.processed_actions, joint_ids=self._joint_ids)

class NavigationAction(JointAction):
    """Joint action term that applies the processed actions as velocity commands."""

    cfg: actions_cfg.NavigationActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.NavigationActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if self.cfg.command_type not in ["vel", "pos", "acc"]:
            raise ValueError(f"Unsupported command_type {self.cfg.command_type}. Supported types are 'vel', 'pos', 'acc'.")
        elif self.cfg.command_type == "pos":
            raise NotImplementedError("Position command type is not implemented yet.")
        elif self.cfg.command_type == "vel":
            pass
        elif self.cfg.command_type == "acc":
            raise NotImplementedError("Acceleration command type is not implemented yet.")
        
        
    def apply_actions(self):
        # set joint navigation targets
        self._asset.set_navigation_target(self.processed_actions, joint_ids=self._joint_ids)