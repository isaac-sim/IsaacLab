# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import Callable, ClassVar, Iterable, Optional, Union

from omni.isaac.orbit.utils import configclass


@configclass
class BaseActuatorCfg:
    """Base configuration for actuator model.

    Note:
        This class is not meant to be instantiated directly, i.e., it should
        only by used as a base class for other actuator model configurations.
    """

    cls_name: ClassVar[Optional[str]] = MISSING
    """
    Name of the associated actuator class.

    This is used to construct the actuator model. If an "implicit" model, then the class name
    is :obj:`None`. Otherwise, it is the name of the actuator model class.
    """

    model_type: ClassVar[str] = MISSING
    """Type of actuator model: "explicit" or "implicit"."""


"""
Actuator model configurations.
"""


@configclass
class ImplicitActuatorCfg(BaseActuatorCfg):
    """Configuration for implicit actuator model.

    Note:
        There are no classes associated to this model. It is handled directly by the physics
        simulation. The provided configuration is used to set the physics simulation parameters.
    """

    cls_name: ClassVar[Optional[str]] = None  # implicit actuators are handled directly.
    model_type: ClassVar[str] = "implicit"

    torque_limit: Optional[float] = None
    """
    Torque saturation (in N-m). Defaults to :obj:`None`.

    This is used by the physics simulation. If :obj:`None`, then default values from USD is used.
    """

    # TODO (@mayank): Check if we can remove this parameter and use the default values?
    velocity_limit: Optional[float] = None
    """
    Velocity saturation (in rad/s or m/s). Defaults to :obj:`None`.

    This is used by the physics simulation. If :obj:`None`, then default values from USD is used.

    Tip:
        Setting this parameter low may result in undesired behaviors. Keep it high in general.
    """


@configclass
class IdealActuatorCfg(BaseActuatorCfg):
    """Configuration for ideal actuator model."""

    cls_name: ClassVar[str] = "IdealActuator"
    model_type: ClassVar[str] = "explicit"

    motor_torque_limit: float = MISSING
    """Effort limit on the motor controlling the actuator (in N-m)."""

    gear_ratio: float = 1.0
    """The gear ratio of the gear box from motor to joint axel. Defaults to 1.0."""


@configclass
class DCMotorCfg(IdealActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    cls_name: ClassVar[str] = "DCMotor"

    peak_motor_torque: float = MISSING
    """Peak motor torque of the electric DC motor (in N-m)."""

    motor_velocity_limit: float = MISSING
    """Maximum velocity of the motor controlling the actuated joint (in rad/s)."""


@configclass
class VariableGearRatioDCMotorCfg(DCMotorCfg):
    """Configuration for variable gear-ratio DC motor actuator model."""

    cls_name: ClassVar[str] = "VariableGearRatioDCMotor"

    # gear ratio is a function of dof positions
    gear_ratio: Union[str, Callable[[torch.Tensor], torch.Tensor]] = MISSING
    """
    Gear ratio function of the gear box connecting the motor to actuated joint.

    Note:
        The gear ratio function takes as input the joint positions.
    """


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""

    cls_name: ClassVar[str] = "ActuatorNetMLP"

    network_file: str = MISSING
    """Path to the file containing network weights."""

    pos_scale: float = MISSING  # DOF position input scaling
    """Scaling of the joint position errors input to the network."""
    vel_scale: float = MISSING  # DOF velocity input scaling
    """Scaling of the joint velocities input to the network."""
    torque_scale: float = MISSING  # DOF torque output scaling
    """Scaling of the joint efforts output from the network."""
    input_idx: Iterable[int] = MISSING  # Indices from the actuator history buffer to pass as inputs.
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    cls_name: ClassVar[str] = "ActuatorNetLSTM"

    network_file: str = MISSING
    """Path to the file containing network weights."""
