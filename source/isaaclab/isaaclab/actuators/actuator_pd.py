# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from isaaclab.utils import DelayBuffer, LinearInterpolation
from isaaclab.utils.types import ArticulationActions

from ._compat import _effort_limits_equal, _resolve_limit_aliases
from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_control import ActuatorControl
    from .actuator_pd_cfg import (
        DCMotorCfg,
        DelayedPDActuatorCfg,
        IdealPDActuatorCfg,
        ImplicitActuatorCfg,
        RemotizedPDActuatorCfg,
    )

# import logger
logger = logging.getLogger(__name__)

"""
Implicit Actuator Models.
"""


class ImplicitActuator(ActuatorBase):
    """Implicit actuator model that is handled by the simulation.

    The articulation writes the configured gains and solver limits to the
    backend, whose discrete solver applies the joint drive. This model also
    computes approximate effort telemetry from the current state because the
    solver does not expose the applied joint effort on every backend.
    """

    cfg: ImplicitActuatorCfg
    """The configuration for the actuator model."""

    is_implicit_model: ClassVar[bool] = True

    #@classmethod
    #def _build_execution_actuator(
    #    cls, actuators: Sequence[ImplicitActuator], joint_indices: torch.Tensor
    #) -> ImplicitActuator:
    #    """Build one private executor for a shared implicit execution batch.

    #    Retains the first group's config metadata; replaces the execution
    #    tensors below without cloning the logical groups' tensor storage.
    #    """
    #    executor = copy.copy(actuators[0])
    #    executor._joint_names = [name for actuator in actuators for name in actuator.joint_names]
    #    executor._joint_indices = joint_indices
    #    executor.velocity_limit = torch.cat([actuator.velocity_limit for actuator in actuators], dim=1)
    #    executor.computed_effort = torch.zeros(executor._num_envs, len(executor._joint_names), device=executor._device)
    #    executor.applied_effort = torch.zeros_like(executor.computed_effort)
    #    return executor

    def __init__(
        self,
        cfg: ImplicitActuatorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        actuator_effort_limit: torch.Tensor | float | None = None,
        actuator_velocity_limit: torch.Tensor | float = torch.inf,
        effort_limit: torch.Tensor | float | None = None, # TODO: Deprecated. Remove in 4.0.
        velocity_limit: torch.Tensor | float = torch.inf, # TODO: Deprecated. Remove in 4.0.
    ):
        # call the base class, sets both actuator level limits to infinity. Note the implict actuator gets its effort
        # limit from the articulation joint effort limit.
        super().__init__(cfg, joint_names, joint_ids, num_envs, device, torch.inf, torch.inf)
        self.stiffness = self._parse_joint_parameter(cfg.stiffness, stiffness)
        self.damping = self._parse_joint_parameter(cfg.damping, damping)

        # Simulation-binded articulation joint parameters.
        self._stiffness = None
        self._damping = None
        self._joint_effort_limit = None

    @property
    def stiffness(self) -> torch.Tensor:
        return self._stiffness[:, self.joint_indices]

    @stiffness.setter
    def stiffness(self, value: Any | None) -> None:
        warnings.warn(
            "Setting the stiffness of a implicit actuator through this setter is not supported. Use the articulation's setter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    @property
    def damping(self) -> torch.Tensor:
        return self._damping[:, self.joint_indices]

    @damping.setter
    def damping(self, value: Any | None) -> None:
        warnings.warn(
            "Setting the damping of a implicit actuator through this setter is not supported. Use the articulation's setter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    @property
    def actuator_effort_limit(self) -> torch.Tensor:
        return self._joint_effort_limit[:, self.joint_indices]
    
    @actuator_effort_limit.setter
    def effort_limit(self, value: Any | None) -> None:
        warnings.warn(
            "Setting the actuator_effort_limit of a implicit actuator through this setter is not supported. Use the articulation's setter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    """
    Operations.
    """

    def reset(self, *args, **kwargs):
        # This is a no-op. There is no state to reset for implicit actuators.
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        For an implicit actuator, the desired control action is returned unchanged because the
        physics solver applies the PD drive. This method still computes approximate computed and
        applied effort telemetry from the current joint state. That telemetry may differ from the
        effort applied internally by the solver.

        Args:
            control_action: Desired joint positions [m or rad, depending on joint type], velocities [m/s or rad/s,
                depending on joint type], and feed-forward efforts [N or N·m, depending on joint type].
            joint_pos: Current joint positions [m or rad, depending on joint type], shape
                ``(num_envs, num_joints)``.
            joint_vel: Current joint velocities [m/s or rad/s, depending on joint type], shape
                ``(num_envs, num_joints)``.

        Returns:
            Desired joint positions [m or rad, depending on joint type], velocities [m/s or rad/s, depending on
            joint type], and efforts [N or N·m, depending on joint type].
        """
        # store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self._stiffness[:, self.joint_indices] * error_pos + self._damping[:, self.joint_indices] * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip telemetry using the articulation joint effort limit."""
        return torch.clip(effort, min=-self._joint_effort_limit[:, self.joint_indices], max=self._joint_effort_limit[:, self.joint_indices])

    def _bind_actuator_parameters(self, control: ActuatorControl) -> None:
        """Bind the actuator parameters to the control object. Stores references to the articulation parameters.

        .. note:: We do not slice because if the indices are not contiguous, then the slicing will create a new tensor,
        and we would lose the reference to the original tensor. By keeping the reference, we avoid synchronization
        issues with the articulation parameters, and the batched executor parameters.
        """
        self._stiffness = control.joint_stiffness.torch
        self._damping = control.joint_damping.torch
        self._joint_effort_limit = control.joint_effort_limits.torch


"""
Explicit Actuator Models.
"""


class IdealPDActuator(ActuatorBase):
    r"""Ideal torque-controlled actuator model with a simple saturation model.

    It employs the following model for computing torques for the actuated joint :math:`j`:

    .. math::

        \tau_{j, computed} = k_p * (q_{des} - q) + k_d * (\dot{q}_{des} - \dot{q}) + \tau_{ff}

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

    actuator_effort_limit: torch.Tensor
    """Actuator-model effort clipping limit [N or N·m, depending on joint type].

    Shape is (num_envs, num_joints).
    """

    stiffness: torch.Tensor
    """Actuator stiffness [N/m or N·m/rad, depending on joint type]."""

    damping: torch.Tensor
    """Actuator damping [N·s/m or N·m·s/rad, depending on joint type]."""

    def __init__(
        self,
        cfg: IdealPDActuatorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        actuator_effort_limit: torch.Tensor | float | None = None,
        actuator_velocity_limit: torch.Tensor | float = torch.inf,
        effort_limit: torch.Tensor | float | None = None, # TODO: Deprecated. Remove in 4.0.
        velocity_limit: torch.Tensor | float = torch.inf, # TODO: Deprecated. Remove in 4.0.
    ):
        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            actuator_effort_limit,
            actuator_velocity_limit,
            effort_limit, # TODO: Deprecated. Remove in 4.0.
            velocity_limit, # TODO: Deprecated. Remove in 4.0.
        )
        self.stiffness = self._parse_joint_parameter(cfg.stiffness, stiffness)
        self.damping = self._parse_joint_parameter(cfg.damping, damping)

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
    r"""Direct control (DC) motor actuator model with velocity-based saturation model.

    It uses the same model as the :class:`IdealPDActuator` for computing the torques from input commands.
    However, it implements a saturation model defined by a linear four quadrant DC motor torque-speed curve.

    A DC motor is a type of electric motor that is powered by direct current electricity. In most cases,
    the motor is connected to a constant source of voltage supply, and the current is controlled by a rheostat.
    Depending on various design factors such as windings and materials, the motor can draw a limited maximum power
    from the electronic source, which limits the produced motor torque and speed.

    A DC motor characteristics are defined by the following parameters:

    * No-load speed (:math:`\dot{q}_{motor, max}`) : The maximum-rated speed of the motor at
      zero torque (:attr:`velocity_limit`).
    * Stall torque (:math:`\tau_{motor, stall}`): The maximum-rated torque produced at
      zero speed (:attr:`saturation_effort`).
    * Continuous torque (:math:`\tau_{motor, con}`): The maximum torque that can be outputted for a short period.
      This is often enforced on the current drives for a DC motor to limit overheating, prevent mechanical damage,
      or enforced by electrical limitations (:attr:`effort_limit`).
    * Corner velocity (:math:`V_{c}`): The velocity where the torque-speed curve intersects with continuous torque.

    Based on these parameters, the instantaneous minimum and maximum torques for velocities between corner velocities
    (where torque-speed curve intersects with continuous torque) are defined as follows:

    .. math::

        \tau_{j, max}(\dot{q}) & = clip \left (\tau_{j, stall} \times \left(1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), -∞, \tau_{j, con} \right) \\
        \tau_{j, min}(\dot{q}) & = clip \left (\tau_{j, stall} \times \left( -1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), - \tau_{j, con}, ∞ \right)

    where :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    :math:`\dot{q}_{j, max} = \gamma^{-1} \times  \dot{q}_{motor, max}`, :math:`\tau_{j, con} =
    \gamma \times \tau_{motor, con}` and :math:`\tau_{j, stall} = \gamma \times \tau_{motor, stall}`
    are the maximum joint velocity, continuous joint torque and stall torque, respectively. These parameters
    are read from the configuration instance passed to the class.

    Using these values, the computed torques are clipped to the minimum and maximum values based on the
    instantaneous joint velocity:

    .. math::

        \tau_{j, applied} = clip(\tau_{computed}, \tau_{j, min}(\dot{q}), \tau_{j, max}(\dot{q}))

    If the velocity of the joint is outside corner velocities (this would be due to external forces) the
    applied output torque will be driven to the continuous torque
    (:attr:`actuator_effort_limit`).

    The figure below demonstrates the clipping action for example (velocity, torque) pairs.

    .. figure:: ../../_static/actuator-group/dc_motor_clipping.jpg
        :align: center
        :figwidth: 100%
        :alt: The effort clipping as a function of joint velocity for a linear DC Motor.

    """

    cfg: DCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.saturation_effort is None:
            raise ValueError("The saturation_effort must be provided for the DC motor actuator model.")
        self._saturation_effort = self.cfg.saturation_effort
        # Find the velocity where the torque-speed curve intersects actuator_effort_limit.
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.actuator_effort_limit / self._saturation_effort)
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
        # save current joint vel
        self._joint_vel[:] = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
        # compute torque limits
        torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        torque_speed_bottom = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        # -- max limit
        max_effort = torch.clip(torque_speed_top, max=self.actuator_effort_limit)
        # -- min limit
        min_effort = torch.clip(torque_speed_bottom, min=-self.actuator_effort_limit)
        # clip the torques based on the motor limits
        clamped = torch.clip(effort, min=min_effort, max=max_effort)
        return clamped


class DelayedPDActuator(IdealPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
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
    """Ideal PD actuator with angle-dependent torque limits.

    This class extends the :class:`DelayedPDActuator` class by adding angle-dependent torque limits to the actuator.
    The torque limits are applied by querying a lookup table describing the relationship between the joint angle
    and the maximum output torque. The lookup table is provided in the configuration instance passed to the class.

    The torque limits are interpolated based on the current joint positions and applied to the actuator commands.
    """

    def __init__(
        self,
        cfg: RemotizedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        actuator_effort_limit: torch.Tensor | float | None = None,
        actuator_velocity_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf, # TODO: Deprecated. Remove in 4.0.
        effort_limit: torch.Tensor | float | None = None, # TODO: Deprecated. Remove in 4.0.
    ):
        if cfg.effort_limit is not None or cfg.effort_limit_sim is not None or cfg.velocity_limit_sim is not None:
            _resolve_limit_aliases(type(self).__name__, cfg, joint_names)
        # remove effort and velocity box constraints from the base class
        cfg.actuator_effort_limit = torch.inf
        cfg.velocity_limit = torch.inf
        # Call the base method with unbounded model clipping and velocity limits.
        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            actuator_effort_limit,
            actuator_velocity_limit,
            effort_limit, # TODO: Deprecated. Remove in 4.0.
            velocity_limit, # TODO: Deprecated. Remove in 4.0.
        )
        self._joint_parameter_lookup = torch.tensor(cfg.joint_parameter_lookup, device=device)
        # define remotized joint torque limit
        self._torque_limit = LinearInterpolation(self.angle_samples, self.max_torque_samples, device=device)

    """
    Properties.
    """

    @property
    def angle_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 0]

    @property
    def transmission_ratio_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 1]

    @property
    def max_torque_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 2]

    """
    Operations.
    """

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
