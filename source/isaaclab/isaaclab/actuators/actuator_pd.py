# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from isaaclab.utils import DelayBuffer, LinearInterpolation
from isaaclab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase, _effort_limits_equal
from .actuator_base_cfg import _resolve_limit_aliases

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


def _initialize_pd_gains(
    actuator: ImplicitActuator | IdealPDActuator,
    stiffness: torch.Tensor | float,
    damping: torch.Tensor | float,
) -> None:
    """Resolve stiffness and damping for a PD actuator model."""
    for parameter_name, default_value in (("stiffness", stiffness), ("damping", damping)):
        cfg_value = getattr(actuator.cfg, parameter_name)
        value = actuator._parse_joint_parameter(cfg_value, default_value)
        setattr(actuator, parameter_name, value)


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

    stiffness: torch.Tensor
    """Live articulation joint stiffness [N/m or N·m/rad, depending on joint type]."""

    damping: torch.Tensor
    """Live articulation joint damping [N·s/m or N·m·s/rad, depending on joint type]."""

    class _JointDrive:
        """Live projection of articulation-owned implicit joint-drive properties."""

        def __init__(self, control: ActuatorControl, joint_indices: slice | torch.Tensor):
            self._control = control
            self._joint_indices = joint_indices

        @property
        def stiffness(self) -> torch.Tensor:
            return self._control.joint_stiffness.torch[:, self._joint_indices]

        @property
        def damping(self) -> torch.Tensor:
            return self._control.joint_damping.torch[:, self._joint_indices]

        @property
        def joint_effort_limit(self) -> torch.Tensor:
            return self._control.joint_effort_limits.torch[:, self._joint_indices]

    @classmethod
    def _build_execution_actuator(
        cls, actuators: Sequence[ImplicitActuator], joint_indices: torch.Tensor
    ) -> ImplicitActuator:
        """Build one private executor for a shared implicit execution batch.

        Retains the first group's config metadata; replaces the execution
        tensors below without cloning the logical groups' tensor storage.
        """
        executor = copy.copy(actuators[0])
        executor._joint_names = [name for actuator in actuators for name in actuator.joint_names]
        executor._joint_indices = joint_indices
        executor.velocity_limit = torch.cat([actuator.velocity_limit for actuator in actuators], dim=1)
        executor.computed_effort = torch.zeros(executor._num_envs, len(executor._joint_names), device=executor._device)
        executor.applied_effort = torch.zeros_like(executor.computed_effort)
        return executor

    @property
    def stiffness(self) -> torch.Tensor:
        """Current joint stiffness values [N/m or N·m/rad, depending on joint type]."""
        joint_drive = self.__dict__.get("_joint_drive")
        return self._stiffness if joint_drive is None else joint_drive.stiffness

    @stiffness.setter
    def stiffness(self, value: torch.Tensor) -> None:
        self._set_joint_drive_property("stiffness", value)

    @property
    def damping(self) -> torch.Tensor:
        """Current joint damping values [N·s/m or N·m·s/rad, depending on joint type]."""
        joint_drive = self.__dict__.get("_joint_drive")
        return self._damping if joint_drive is None else joint_drive.damping

    @damping.setter
    def damping(self, value: torch.Tensor) -> None:
        self._set_joint_drive_property("damping", value)

    @property
    def joint_effort_limit(self) -> torch.Tensor:
        """Current joint effort limits [N or N·m, depending on joint type]."""
        joint_drive = self.__dict__.get("_joint_drive")
        return self._joint_effort_limit if joint_drive is None else joint_drive.joint_effort_limit

    @joint_effort_limit.setter
    def joint_effort_limit(self, value: torch.Tensor) -> None:
        self._set_joint_drive_property("joint_effort_limit", value)

    @property
    def effort_limit(self) -> torch.Tensor:
        """Deprecated joint effort limit [N or N·m, depending on joint type].

        .. deprecated:: 3.0
            Use :attr:`joint_effort_limit` instead. This alias will be removed in 4.0.
        """
        warnings.warn(
            "ImplicitActuator.effort_limit is deprecated. Use joint_effort_limit instead; "
            "effort_limit will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_effort_limit

    @effort_limit.setter
    def effort_limit(self, value: torch.Tensor) -> None:
        warnings.warn(
            "ImplicitActuator.effort_limit is deprecated. Use joint_effort_limit instead; "
            "effort_limit will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.joint_effort_limit = value

    def _set_joint_drive_property(self, name: str, value: torch.Tensor) -> None:
        """Store a construction value or reject assignment after articulation binding."""
        if "_joint_drive" in self.__dict__:
            writer_name = {
                "stiffness": "write_joint_stiffness_to_sim_index",
                "damping": "write_joint_damping_to_sim_index",
                "joint_effort_limit": "write_joint_effort_limit_to_sim_index",
            }[name]
            raise AttributeError(
                f"ImplicitActuator.{name} is articulation-owned after binding. Use "
                f"Articulation.{writer_name}() or randomize_actuator_gains() to update it."
            )
        self.__dict__[f"_{name}"] = value

    def __init__(
        self,
        cfg: ImplicitActuatorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        joint_effort_limit: torch.Tensor | float | None = None,
        velocity_limit: torch.Tensor | float = torch.inf,
        effort_limit: torch.Tensor | float | None = None,
    ):
        if (
            cfg.actuator_effort_limit is not None
            or cfg.effort_limit is not None
            or cfg.effort_limit_sim is not None
            or cfg.velocity_limit_sim is not None
        ):
            _resolve_limit_aliases(type(self).__name__, cfg, joint_names)
        if effort_limit is not None:
            warnings.warn(
                "The effort_limit constructor argument is deprecated. Use joint_effort_limit instead; "
                "effort_limit will be removed in 4.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if joint_effort_limit is not None and not _effort_limits_equal(joint_effort_limit, effort_limit):
                raise ValueError(
                    "Received conflicting joint_effort_limit and deprecated effort_limit constructor arguments."
                )
            joint_effort_limit = effort_limit
        elif joint_effort_limit is None:
            joint_effort_limit = torch.inf
        # velocity limits
        # 'velocity_limit' is the joint's peak velocity (the actuator's rated speed
        # reflected at the joint): it feeds the data buffers
        # (:attr:`ArticulationData.soft_joint_vel_limits`, read by e.g. the
        # ``joint_vel_out_of_limit`` termination) but is NOT pushed to the physics
        # solver. 'joint_velocity_limit' is a solver-level hard clamp (PhysX
        # ``maxJointVelocity``) with no physical counterpart -- a physical actuator
        # limits joint speed through its torque curve, not a kinematic clamp.
        # ``_resolve_limit_aliases`` mirrors a deprecated 'velocity_limit_sim' into
        # 'velocity_limit' so the data buffers stay meaningful.
        if cfg.velocity_limit is not None:
            # notify about the behavior change: this value used to be ignored for implicit actuators
            logger.warning(
                "The <ImplicitActuatorCfg> object has a value for 'velocity_limit'. Previously, this value"
                " was ignored for implicit actuators. It now populates the joint velocity-limit data buffers"
                " (e.g. 'soft_joint_vel_limits' used by velocity-limit terminations and rewards), but it is"
                " still not pushed to the physics solver. To set a solver-level velocity clamp, please use"
                " 'joint_velocity_limit'."
            )

        # call the base class
        super().__init__(cfg, joint_names, joint_ids, num_envs, device, torch.inf, velocity_limit)
        self.joint_effort_limit = self._parse_joint_parameter(cfg.joint_effort_limit, joint_effort_limit)
        _initialize_pd_gains(self, stiffness, damping)

    def _bind_joint_drive(self, control: ActuatorControl) -> None:
        """Bind implicit drive reads to live articulation-owned joint properties."""
        self._joint_drive = self._JointDrive(control, self.joint_indices)
        self.__dict__.pop("_stiffness", None)
        self.__dict__.pop("_damping", None)
        self.__dict__.pop("_joint_effort_limit", None)

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
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip telemetry using the live articulation joint effort limit."""
        return torch.clip(effort, min=-self.joint_effort_limit, max=self.joint_effort_limit)


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

    The model clips the resulting joint effort directly to
    :attr:`actuator_effort_limit`:

    .. math::

        \tau_{j, applied} = clip(\tau_{j, computed}, -\tau_{max}, \tau_{max})

    where :math:`\tau_{max}` is the configured joint-side effort limit [N or
    N·m, depending on joint type].
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

    class _NativeActuatorGains:
        """Live controller-owned gain projection for one native actuator group."""

        def __init__(self, control: ActuatorControl, joint_indices: slice | torch.Tensor):
            self._control = control
            self._joint_indices = joint_indices

        def get(self, attr: Literal["kp", "kd"]) -> torch.Tensor | None:
            """Read one controller gain in the group's public joint order."""
            return self._control.get_native_actuator_gain(attr, self._joint_indices)

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
        velocity_limit: torch.Tensor | float = torch.inf,
        effort_limit: torch.Tensor | float | None = None,
    ):
        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            actuator_effort_limit,
            velocity_limit,
            effort_limit,
        )
        _initialize_pd_gains(self, stiffness, damping)

    @property
    def stiffness(self) -> torch.Tensor:
        """Current actuator stiffness [N/m or N·m/rad, depending on joint type]."""
        native_gains = self.__dict__.get("_native_actuator_gains")
        return self._stiffness if native_gains is None else native_gains.get("kp")

    @stiffness.setter
    def stiffness(self, value: torch.Tensor) -> None:
        self._set_actuator_gain_property("stiffness", value)

    @property
    def damping(self) -> torch.Tensor:
        """Current actuator damping [N·s/m or N·m·s/rad, depending on joint type]."""
        native_gains = self.__dict__.get("_native_actuator_gains")
        return self._damping if native_gains is None else native_gains.get("kd")

    @damping.setter
    def damping(self, value: torch.Tensor) -> None:
        self._set_actuator_gain_property("damping", value)

    def _bind_native_actuator_gains(self, control: ActuatorControl) -> None:
        """Bind native gain reads when controllers cover every joint in this group."""
        native_gains = self._NativeActuatorGains(control, self.joint_indices)
        if native_gains.get("kp") is None or native_gains.get("kd") is None:
            return
        self._native_actuator_gains = native_gains
        self.__dict__.pop("_stiffness", None)
        self.__dict__.pop("_damping", None)

    def _set_actuator_gain_property(self, name: Literal["stiffness", "damping"], value: torch.Tensor) -> None:
        """Store a construction gain or reject assignment after native binding."""
        if "_native_actuator_gains" in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is controller-owned after native binding. Use "
                "randomize_actuator_gains() or the backend native gain API to update it."
            )
        self.__dict__[f"_{name}"] = value

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

    * No-load speed (:math:`\dot{q}_{motor, max}`) [m/s or rad/s, depending on
      joint type]: The maximum-rated speed of the motor at zero torque
      (:attr:`velocity_limit`).
    * Stall torque (:math:`\tau_{motor, stall}`): The maximum-rated torque produced at
      zero speed [N or N·m, depending on joint type] (:attr:`saturation_effort`).
    * Continuous torque (:math:`\tau_{motor, con}`) [N or N·m, depending on
      joint type]: The maximum torque that can be outputted for a short period.
      This is often enforced on the current drives for a DC motor to limit
      overheating, prevent mechanical damage, or enforced by electrical
      limitations (:attr:`actuator_effort_limit`).
    * Corner velocity (:math:`V_{c}`) [m/s or rad/s, depending on joint type]:
      The velocity where the torque-speed curve intersects with continuous torque.

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

    This class extends :class:`DelayedPDActuator` with angle-dependent effort
    limits [N or N·m, depending on joint type]. The limits are applied by
    querying a lookup table describing the relationship between joint angle
    [m or rad, depending on joint type] and maximum output effort [N or N·m,
    depending on joint type]. The lookup table is provided in the configuration
    instance passed to the class.

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
        velocity_limit: torch.Tensor | float = torch.inf,
        effort_limit: torch.Tensor | float | None = None,
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
            velocity_limit,
            effort_limit,
        )
        self._joint_parameter_lookup = torch.tensor(cfg.joint_parameter_lookup, device=device)
        # define remotized joint torque limit
        self._torque_limit = LinearInterpolation(self.angle_samples, self.max_torque_samples, device=device)

    """
    Properties.
    """

    @property
    def angle_samples(self) -> torch.Tensor:
        """Lookup joint positions [m or rad, depending on joint type]."""
        return self._joint_parameter_lookup[:, 0]

    @property
    def transmission_ratio_samples(self) -> torch.Tensor:
        """Dimensionless lookup transmission ratios."""
        return self._joint_parameter_lookup[:, 1]

    @property
    def max_torque_samples(self) -> torch.Tensor:
        """Lookup effort limits [N or N·m, depending on joint type]."""
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
