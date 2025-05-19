# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import IdealPDActuator
from isaaclab.utils import DelayBuffer, LinearInterpolation
from isaacsim.core.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_spot_cfg import DelayedPDActuatorCfg, RemotizedPDActuatorCfg


"""
Spot Actuator Models.
"""


class DelayedPDActuator(IdealPDActuator):
    def __init__(
        self,
        cfg: DelayedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            armature,
            friction,
            effort_limit,
            velocity_limit,
        )
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_num_time_lags, batch_size=num_envs, device=device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_num_time_lags, batch_size=num_envs, device=device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_num_time_lags, batch_size=num_envs, device=device)
        # all of the envs
        self._ALL_INDICES = torch.arange(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        env_size = self._ALL_INDICES[env_ids].size()
        # set a new random delay for environments in env_ids
        time_lags = self.positions_delay_buffer.time_lags
        time_lags[env_ids] = torch.randint(
            low=self.cfg.min_num_time_lags,
            high=self.cfg.max_num_time_lags + 1,
            size=env_size,
            device=self._device,
            dtype=torch.int,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags)
        self.velocities_delay_buffer.set_time_lag(time_lags)
        self.efforts_delay_buffer.set_time_lag(time_lags)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


class RemotizedPDActuator(DelayedPDActuator):
    def __init__(
        self,
        cfg: RemotizedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        # remove effort and velocity box constraints from the base class
        cfg.effort_limit = torch.inf
        cfg.velocity_limit = torch.inf
        # call the base method and set default effort_limit and velocity_limit to inf
        super().__init__(
            cfg, joint_names, joint_ids, num_envs, device, stiffness, damping, armature, friction, torch.inf, torch.inf
        )
        # data: knee angle (rad), transmission ratio (in/out), max output torque (N*m)
        self._data = cfg.data.to(device=device)
        # define remotized joint torque limit
        self._torque_limit = LinearInterpolation(self.angle_samples, self.max_torque_samples, device=device)

    @property
    def angle_samples(self) -> torch.Tensor:
        return self._data[:, 0]

    @property
    def transmission_ratio_samples(self) -> torch.Tensor:
        return self._data[:, 1]

    @property
    def max_torque_samples(self) -> torch.Tensor:
        return self._data[:, 2]

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)
        # compute the absolute torque limits for the current joint positions
        abs_torque_limits = self._torque_limit.compute(joint_pos)
        # apply the limits
        control_action.joint_efforts = torch.clamp(
            control_action.joint_efforts, min=-abs_torque_limits, max=abs_torque_limits
        )
        return control_action
