# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from .joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

from isaaclab.controllers.lee_velocity_control import LeeVelController

class ThrustAction(JointAction):
    """Joint action term that applies the processed actions as thrust commands."""

    cfg: actions_cfg.ThrustActionCfg
    """The configuration of the action term."""
    

    def __init__(self, cfg: actions_cfg.ThrustActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint thrust targets
        # TODO still inherits Articulation instead of ArticulationWithThrusters as default. Is overwritten so doesnt matter but gives ugly warnings in VSCode. Fix in future.
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
        
        self._lvc = LeeVelController(cfg=self.cfg.controller_cfg, asset=self._asset, num_envs=self.num_envs, device=self.device)
        
        
    def apply_actions(self):
        # set joint navigation targets
        # self.processed_actions[:] = torch.clamp(self.processed_actions, min=-1.0, max=1.0)

        clamped_action = torch.clamp(self.processed_actions, min=-1.0, max=1.0)
        max_speed = 2.0  # [m/s]
        max_yawrate = torch.pi / 3.0  # [rad/s]
        max_inclination_angle = torch.pi / 4.0  # [rad]
        
        clamped_action[:, 0] = max_speed * torch.clamp(clamped_action[:, 0], min=0.0, max=1.0)  # only allow positive thrust commands [0, 1]

        self.processed_actions[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        self.processed_actions[:, 1] = 0.0  # set lateral thrust command to 0
        self.processed_actions[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        self.processed_actions[:, 3] = clamped_action[:, 2] * max_yawrate

        wrench_command = self._lvc.compute(self.processed_actions)
        thrust_commands = (torch.pinverse(self._asset._allocation_matrix) @ wrench_command.T).T
        self._asset.set_thrust_target(thrust_commands, joint_ids=self._joint_ids)