# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import DCMotor

if TYPE_CHECKING:
    from .actuator_cfg import IdentifiedActuatorCfg


class IdentifiedActuator(DCMotor):
    cfg: IdentifiedActuatorCfg

    def __init__(self, cfg: IdentifiedActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.)

    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model on the torque
        control_action.joint_efforts = control_action.joint_efforts - (self.friction_static * torch.tanh(
            joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel)

        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
