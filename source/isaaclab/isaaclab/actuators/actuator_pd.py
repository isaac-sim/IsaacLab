# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.utils import DelayBuffer, LinearInterpolation
from isaaclab.utils.types import ArticulationActions

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

    The articulation class sets the stiffness and damping parameters from the implicit actuator configuration
    into the simulation. Thus, the class does not perform its own computations on the joint action that
    needs to be applied to the simulation. However, it computes the approximate torques for the actuated joint
    since PhysX does not expose this quantity explicitly.

    .. caution::

        The class is only provided for consistency with the other actuator models. It does not implement any
        functionality and should not be used. All values should be set to the simulation directly.
    """

    cfg: ImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: ImplicitActuatorCfg, *args, **kwargs):
        # effort limits
        if cfg.effort_limit_sim is None and cfg.effort_limit is not None:
            # throw a warning that we have a replacement for the deprecated parameter
            omni.log.warn(
                "The <ImplicitActuatorCfg> object has a value for 'effort_limit'."
                " This parameter will be removed in the future."
                " To set the effort limit, please use 'effort_limit_sim' instead."
            )
            cfg.effort_limit_sim = cfg.effort_limit
        elif cfg.effort_limit_sim is not None and cfg.effort_limit is None:
            # TODO: Eventually we want to get rid of 'effort_limit' for implicit actuators.
            #   We should do this once all parameters have an "_sim" suffix.
            cfg.effort_limit = cfg.effort_limit_sim
        elif cfg.effort_limit_sim is not None and cfg.effort_limit is not None:
            if cfg.effort_limit_sim != cfg.effort_limit:
                raise ValueError(
                    "The <ImplicitActuatorCfg> object has set both 'effort_limit_sim' and 'effort_limit'"
                    f" and they have different values {cfg.effort_limit_sim} != {cfg.effort_limit}."
                    " Please only set 'effort_limit_sim' for implicit actuators."
                )

        # velocity limits
        if cfg.velocity_limit_sim is None and cfg.velocity_limit is not None:
            # throw a warning that previously this was not set
            # it leads to different simulation behavior so we want to remain backwards compatible
            omni.log.warn(
                "The <ImplicitActuatorCfg> object has a value for 'velocity_limit'."
                " Previously, although this value was specified, it was not getting used by implicit"
                " actuators. Since this parameter affects the simulation behavior, we continue to not"
                " use it. This parameter will be removed in the future."
                " To set the velocity limit, please use 'velocity_limit_sim' instead."
            )
            cfg.velocity_limit = None
        elif cfg.velocity_limit_sim is not None and cfg.velocity_limit is None:
            # TODO: Eventually we want to get rid of 'velocity_limit' for implicit actuators.
            #   We should do this once all parameters have an "_sim" suffix.
            cfg.velocity_limit = cfg.velocity_limit_sim
        elif cfg.velocity_limit_sim is not None and cfg.velocity_limit is not None:
            if cfg.velocity_limit_sim != cfg.velocity_limit:
                raise ValueError(
                    "The <ImplicitActuatorCfg> object has set both 'velocity_limit_sim' and 'velocity_limit'"
                    f" and they have different values {cfg.velocity_limit_sim} != {cfg.velocity_limit}."
                    " Please only set 'velocity_limit_sim' for implicit actuators."
                )

        # set implicit actuator model flag
        ImplicitActuator.is_implicit_model = True
        # call the base class
        super().__init__(cfg, *args, **kwargs)

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

        In case of implicit actuator, the control action is directly returned as the computed action.
        This function is a no-op and does not perform any computation on the input control action.
        However, it computes the approximate torques for the actuated joint since PhysX does not compute
        this quantity explicitly.

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
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
    r"""Direct control (DC) motor actuator model with velocity-based saturation model.

    It uses the same model as the :class:`IdealPDActuator` for computing the torques from input commands.
    However, it implements a saturation model defined by a linear four quadrant DC motor torque-speed curve.

    A DC motor is a type of electric motor that is powered by direct current electricity. In most cases,
    the motor is connected to a constant source of voltage supply, and the current is controlled by a rheostat.
    Depending on various design factors such as windings and materials, the motor can draw a limited maximum power
    from the electronic source, which limits the produced motor torque and speed.

    A DC motor characteristics are defined by the following parameters:

    * No-load speed (:math:`\dot{q}_{motor, max}`) : The maximum-rated speed of the motor at 0 Torque (:attr:`velocity_limit`).
    * Stall torque (:math:`\tau_{motor, stall}`): The maximum-rated torque produced at 0 speed (:attr:`saturation_effort`).
    * Continuous torque (:math:`\tau_{motor, con}`): The maximum torque that can be outputted for a short period. This
      is often enforced on the current drives for a DC motor to limit overheating, prevent mechanical damage, or
      enforced by electrical limitations.(:attr:`effort_limit`).
    * Corner velocity (:math:`V_{c}`): The velocity where the torque-speed curve intersects with continuous torque.
      Based on these parameters, the instantaneous minimum and maximum torques for velocities between corner velocities
      (where torque-speed curve intersects with continuous torque) are defined as follows:

    Based on these parameters, the instantaneous minimum and maximum torques for velocities are defined as follows:

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
    applied output torque will be driven to the Continuous Torque (`effort_limit`).

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
        # find the velocity on the torque-speed curve that intersects effort_limit in the second and fourth quadrant
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
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
        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        # -- min limit
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)
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
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        # remove effort and velocity box constraints from the base class
        cfg.effort_limit = torch.inf
        cfg.velocity_limit = torch.inf
        # call the base method and set default effort_limit and velocity_limit to inf
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
            dynamic_friction,
            viscous_friction,
            effort_limit,
            velocity_limit,
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
