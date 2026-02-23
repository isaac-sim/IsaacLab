# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils.warp.update_kernels import update_array2D_with_array2D_masked

from .actuator_base import ActuatorBase
from .kernels import clip_efforts_dc_motor, compute_pd_actuator

# from isaaclab.utils import DelayBuffer, LinearInterpolation


if TYPE_CHECKING:
    from isaaclab.actuators.actuator_cfg import (  # DelayedPDActuatorCfg,; RemotizedPDActuatorCfg,
        DCMotorCfg,
        IdealPDActuatorCfg,
        ImplicitActuatorCfg,
    )

# import logger
logger = logging.getLogger(__name__)


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
            logger.warning(
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
            logger.warning(
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

    def compute(self):
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
        wp.launch(
            compute_pd_actuator,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                self.data._actuator_position_target,
                self.data._actuator_velocity_target,
                self.data._actuator_effort_target,
                self.data._sim_bind_joint_pos,
                self.data._sim_bind_joint_vel,
                self.data._actuator_stiffness,
                self.data._actuator_damping,
                self.data._computed_effort,
                self._env_mask,
                self._joint_mask,
            ],
            device=self._device,
        )
        self._clip_effort(self.data._computed_effort, self.data._applied_effort)
        # update the joint effort
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self._num_envs, self.num_joints),
            inputs=[
                self.data._actuator_effort_target,
                self.data.joint_effort,
                self._env_mask,
                self._joint_mask,
            ],
            device=self._device,
        )


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

    def compute(self):
        wp.launch(
            compute_pd_actuator,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                self.data._actuator_position_target,
                self.data._actuator_velocity_target,
                self.data._actuator_effort_target,
                self.data._sim_bind_joint_pos,
                self.data._sim_bind_joint_vel,
                self.data._actuator_stiffness,
                self.data._actuator_damping,
                self.data._computed_effort,
                self._env_mask,
                self._joint_mask,
            ],
            device=self._device,
        )
        self._clip_effort(self.data._computed_effort, self.data._applied_effort)
        # update the joint effort
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self._num_envs, self.num_joints),
            inputs=[
                self.data._applied_effort,
                self.data.joint_effort,
                self._env_mask,
                self._joint_mask,
            ],
            device=self._device,
        )


class DCMotor(IdealPDActuator):
    r"""Direct control (DC) motor actuator model with velocity-based saturation model.

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
            self._saturation_effort = wp.inf
        # check that quantities are provided
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity limit must be provided for the DC motor actuator model.")

    """
    Operations.
    """

    def compute(self):
        # calculate the desired joint torques
        return super().compute()

    """
    Helper functions.
    """

    def _clip_effort(self, effort: wp.array, clipped_effort: wp.array) -> None:
        wp.launch(
            clip_efforts_dc_motor,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                self._saturation_effort,
                self.data._sim_bind_joint_effort_limits_sim,
                self.data._sim_bind_joint_vel_limits_sim,
                self.data._sim_bind_joint_vel,
                effort,
                clipped_effort,
                self._env_mask,
                self._joint_mask,
            ],
            device=self._device,
        )


# class DelayedPDActuator(IdealPDActuator):
#    """Ideal PD actuator with delayed command application.
#
#    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
#    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
#    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
#    applied to the simulation is lagged by a certain number of physics steps.
#
#    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
#    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
#    to the class.
#    """
#
#    cfg: DelayedPDActuatorCfg
#    """The configuration for the actuator model."""
#
#    def __init__(self, cfg: DelayedPDActuatorCfg, *args, **kwargs):
#        super().__init__(cfg, *args, **kwargs)
#        # instantiate the delay buffers
#        self.joint_targets_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
#        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
#        # all of the envs
#        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
#
#    def reset(self, env_ids: Sequence[int]):
#        super().reset(env_ids)
#        # number of environments (since env_ids can be a slice)
#        if env_ids is None or env_ids == slice(None):
#            num_envs = self._num_envs
#        else:
#            num_envs = len(env_ids)
#        # set a new random delay for environments in env_ids
#        time_lags = torch.randint(
#            low=self.cfg.min_delay,
#            high=self.cfg.max_delay + 1,
#            size=(num_envs,),
#            dtype=torch.int,
#            device=self._device,
#        )
#        # set delays
#        self.joint_targets_delay_buffer.set_time_lag(time_lags, env_ids)
#        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
#        # reset buffers
#        self.joint_targets_delay_buffer.reset(env_ids)
#        self.efforts_delay_buffer.reset(env_ids)
#
#    def compute(
#        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
#    ) -> ArticulationActions:
#        # apply delay based on the delay the model for all the setpoints
#        control_action.joint_targets = self.joint_targets_delay_buffer.compute(control_action.joint_targets)
#        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
#        # compte actuator model
#        return super().compute(control_action, joint_pos, joint_vel)
#
#
# class RemotizedPDActuator(DelayedPDActuator):
#    """Ideal PD actuator with angle-dependent torque limits.
#
#    This class extends the :class:`DelayedPDActuator` class by adding angle-dependent torque limits to the actuator.
#    The torque limits are applied by querying a lookup table describing the relationship between the joint angle
#    and the maximum output torque. The lookup table is provided in the configuration instance passed to the class.
#
#    The torque limits are interpolated based on the current joint positions and applied to the actuator commands.
#    """
#
#    def __init__(
#        self,
#        cfg: RemotizedPDActuatorCfg,
#        joint_names: list[str],
#        joint_ids: Sequence[int],
#        num_envs: int,
#        device: str,
#        control_mode: str = "position",
#        stiffness: torch.Tensor | float = 0.0,
#        damping: torch.Tensor | float = 0.0,
#        armature: torch.Tensor | float = 0.0,
#        friction: torch.Tensor | float = 0.0,
#        effort_limit: torch.Tensor | float = torch.inf,
#        velocity_limit: torch.Tensor | float = torch.inf,
#    ):
#        # remove effort and velocity box constraints from the base class
#        cfg.effort_limit = torch.inf
#        cfg.velocity_limit = torch.inf
#        # call the base method and set default effort_limit and velocity_limit to inf
#        super().__init__(
#            cfg,
#            joint_names,
#            joint_ids,
#            num_envs,
#            device,
#            control_mode,
#            stiffness,
#            damping,
#            armature,
#            friction,
#            torch.inf,
#            torch.inf,
#        )
#        self._joint_parameter_lookup = torch.tensor(cfg.joint_parameter_lookup, device=device)
#        # define remotized joint torque limit
#        self._torque_limit = LinearInterpolation(self.angle_samples, self.max_torque_samples, device=device)
#
#    """
#    Properties.
#    """
#
#    @property
#    def angle_samples(self) -> torch.Tensor:
#        return self._joint_parameter_lookup[:, 0]
#
#    @property
#    def transmission_ratio_samples(self) -> torch.Tensor:
#        return self._joint_parameter_lookup[:, 1]
#
#    @property
#    def max_torque_samples(self) -> torch.Tensor:
#        return self._joint_parameter_lookup[:, 2]
#
#    """
#    Operations.
#    """
#
#    def compute(
#        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
#    ) -> ArticulationActions:
#        # call the base method
#        control_action = super().compute(control_action, joint_pos, joint_vel)
#        # compute the absolute torque limits for the current joint positions
#        abs_torque_limits = self._torque_limit.compute(joint_pos)
#        # apply the limits
#        control_action.joint_efforts = torch.clamp(
#            control_action.joint_efforts, min=-abs_torque_limits, max=abs_torque_limits
#        )
#        self.applied_effort = control_action.joint_efforts
#        return control_action
