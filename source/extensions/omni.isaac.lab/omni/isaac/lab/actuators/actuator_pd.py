# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.utils import DelayBuffer, LinearInterpolation

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import (
        DCMotorCfg,
        DelayedPDActuatorCfg,
        IdealPDActuatorCfg,
        ImplicitActuatorCfg,
        RemotizedPDActuatorCfg,
    )


"""
Implicit Actuator Models.
"""


class ImplicitActuator(ActuatorBase):
    """Implicit actuator model that is handled by the simulation.

    This performs a similar function as the :class:`IdealPDActuator` class. However, the PD control is handled
    implicitly by the simulation which performs continuous-time integration of the PD control law. This is
    generally more accurate than the explicit PD control law used in :class:`IdealPDActuator` when the simulation
    time-step is large.

    .. note::

        The articulation class sets the stiffness and damping parameters from the configuration into the simulation.
        Thus, the parameters are not used in this class.

    .. caution::

        The class is only provided for consistency with the other actuator models. It does not implement any
        functionality and should not be used. All values should be set to the simulation directly.
    """

    cfg: ImplicitActuatorCfg
    """The configuration for the actuator model."""

    """
    Operations.
    """

    def reset(self, *args, **kwargs):
        # This is a no-op. There is no state to reset for implicit actuators.
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute the aproximmate torques for the actuated joint (physX does not compute this explicitly)."""
        # store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action


"""
Explicit Actuator Models.
"""


class IdealPDActuator(ActuatorBase):
    r"""Ideal torque-controlled actuator model with a simple saturation model.

    It employs the following model for computing torques for the actuated joint :math:`j`:

    .. math::

        \tau_{j, computed} = k_p * (q - q_{des}) + k_d * (\dot{q} - \dot{q}_{des}) + \tau_{ff}

    where, :math:`k_p` and :math:`k_d` are joint stiffness and damping gains, :math:`q` and :math:`\dot{q}`
    are the current joint positions and velocities, :math:`q_{des}`, :math:`\dot{q}_{des}` and :math:`\tau_{ff}`
    are the desired joint positions, velocities and torques commands.

    The clipping model is based on the maximum torque applied by the motor. It is implemented as:

    .. math::

        \tau_{j, max} & = \gamma \times \tau_{motor, max} \\
        \tau_{j, applied} & = clip(\tau_{computed}, -\tau_{j, max}, \tau_{j, max})

    where the clipping function is defined as :math:`clip(x, x_{min}, x_{max}) = min(max(x, x_{min}), x_{max})`.
    The parameters :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    and :math:`\tau_{motor, max}` is the maximum motor effort possible. These parameters are read from
    the configuration instance passed to the class.
    """

    cfg: IdealPDActuatorCfg
    """The configuration for the actuator model."""

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute errors
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


class DCMotor(IdealPDActuator):
    r"""
    Direct control (DC) motor actuator model with velocity-based saturation model.

    It uses the same model as the :class:`IdealActuator` for computing the torques from input commands.
    However, it implements a saturation model defined by DC motor characteristics.

    A DC motor is a type of electric motor that is powered by direct current electricity. In most cases,
    the motor is connected to a constant source of voltage supply, and the current is controlled by a rheostat.
    Depending on various design factors such as windings and materials, the motor can draw a limited maximum power
    from the electronic source, which limits the produced motor torque and speed.

    A DC motor characteristics are defined by the following parameters:

    * Continuous-rated speed (:math:`\dot{q}_{motor, max}`) : The maximum-rated speed of the motor.
    * Continuous-stall torque (:math:`\tau_{motor, max}`): The maximum-rated torque produced at 0 speed.
    * Saturation torque (:math:`\tau_{motor, sat}`): The maximum torque that can be outputted for a short period.

    Based on these parameters, the instantaneous minimum and maximum torques are defined as follows:

    .. math::

        \tau_{j, max}(\dot{q}) & = clip \left (\tau_{j, sat} \times \left(1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), 0.0, \tau_{j, max} \right) \\
        \tau_{j, min}(\dot{q}) & = clip \left (\tau_{j, sat} \times \left( -1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), - \tau_{j, max}, 0.0 \right)

    where :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    :math:`\dot{q}_{j, max} = \gamma^{-1} \times  \dot{q}_{motor, max}`, :math:`\tau_{j, max} =
    \gamma \times \tau_{motor, max}` and :math:`\tau_{j, peak} = \gamma \times \tau_{motor, peak}`
    are the maximum joint velocity, maximum joint torque and peak torque, respectively. These parameters
    are read from the configuration instance passed to the class.

    Using these values, the computed torques are clipped to the minimum and maximum values based on the
    instantaneous joint velocity:

    .. math::

        \tau_{j, applied} = clip(\tau_{computed}, \tau_{j, min}(\dot{q}), \tau_{j, max}(\dot{q}))

    """

    cfg: DCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.saturation_effort is not None:
            self._saturation_effort = self.cfg.saturation_effort
        else:
            self._saturation_effort = torch.inf
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # check that quantities are provided
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity limit must be provided for the DC motor actuator model.")

    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        return super().compute(control_action, joint_pos, joint_vel)

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # compute torque limits
        # -- max limit
        max_effort = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)
        # -- min limit
        min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)

        # clip the torques based on the motor limits
        return torch.clip(effort, min=min_effort, max=max_effort)


class DelayedPDActuator(IdealPDActuator):
    """Ideal PD actuator with delayed data.

    The DelayedPDActuator has configurable minimum and maximum time lag values, which are used to initialize a
    DelayBuffer to hold a queue of pending actuator commands. On reset, a value time_lags will be randomly sampled
    from the min and max time lag bounds. At every physics step, the most recent actuation value is pushed to the
    DelayBuffer, but the final actuation value applied to simulation will be `time_lags` physics steps in the past.
    """

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
        self.positions_delay_buffer = DelayBuffer(cfg.max_num_time_lags, num_envs=num_envs, device=device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_num_time_lags, num_envs=num_envs, device=device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_num_time_lags, num_envs=num_envs, device=device)
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
    """Ideal PD actuator with angle dependent torque limits.

    The torque limits for this actuator are applied by querying a lookup table describing the relationship between
    the joint angle and the maximum output torque.
    """

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
        self._joint_parameter_lookup = cfg.joint_parameter_lookup.to(device=device)
        # define remotized joint torque limit
        self._torque_limit = LinearInterpolation(self.angle_samples, self.max_torque_samples, device=device)

    @property
    def angle_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 0]

    @property
    def transmission_ratio_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 1]

    @property
    def max_torque_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 2]

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
        self.applied_effort = control_action.joint_efforts
        return control_action
