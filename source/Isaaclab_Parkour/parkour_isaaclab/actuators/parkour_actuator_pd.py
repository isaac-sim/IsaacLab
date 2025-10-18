# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from isaaclab.actuators.actuator_pd import IdealPDActuator
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .parkour_actuator_cfg import ParkourDCMotorCfg


class ParkourDCMotor(IdealPDActuator):
    cfg: ParkourDCMotorCfg

    """
    Actually this motor remove D term, and feedforward term
    reference code: legged_gym 
        torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        JointPoision(use_default_offset = True) -> self.processed_actions == (actions_scaled + self.default_dof_pos) 
    """

    def __init__(self, cfg: ParkourDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if self.cfg.saturation_effort is not None:
            if isinstance(self.cfg.saturation_effort, dict):
                self._saturation_effort = self._parse_joint_parameter(self.cfg.saturation_effort, torch.zeros_like(self.computed_effort))
            else:
                self._saturation_effort = self.cfg.saturation_effort
        else:
            self._saturation_effort = torch.inf
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # check that quantities are provided
        # if self.cfg.velocity_limit is None:
        #     pass 


    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint 
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
    
    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        if self.cfg.saturation_effort is not None:
            max_effort = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
            max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)
            min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
            min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)           
            return torch.clip(effort, min=min_effort, max=max_effort)
        else:
            return super()._clip_effort(effort)

