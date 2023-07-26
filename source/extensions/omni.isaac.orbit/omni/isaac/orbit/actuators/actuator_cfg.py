# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Iterable

from omni.isaac.orbit.utils import configclass

from .actuator import ActuatorBase, DCMotor, IdealPDActuator, ImplicitActuator
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP


@configclass
class ActuatorBaseCfg:
    """Configuration for default actuators in an articulation."""

    cls: type[ActuatorBase] = MISSING
    """Actuator class."""

    dof_name_expr: list[str] = MISSING
    """Articulation's DOF names that are part of the group. Can be regex expressions (e.g. ".*")."""

    effort_limit: float | None = None
    """
    Force/Torque limit of the DOFs in the group. Defaults to :obj:`None`.
    """

    velocity_limit: float | None = None
    """
    Velocity limit of the DOFs in the group. Defaults to :obj:`None`.
    """

    stiffness: dict[str, float] | None = None
    """
    Stiffness gains (Also known as P gain) of the DOFs in the group. Defaults to :obj:`None`.
    If :obj:`None`, these are loaded from the articulation prim.
    """

    damping: dict[str, float] | None = None
    """
    Damping gains (Also known as D gain) of the DOFs in the group. Defaults to :obj:`None`.
    If :obj:`None`, these are loaded from the articulation prim.
    """


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator.
    The PD control is handled implicitly by the simulation."""

    cls: type[ActuatorBase] = IdealPDActuator
    """Actuator class."""


@configclass
class ImplicitPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator.
    The PD control is handled implicitly by the simulation."""

    cls: type[ActuatorBase] = ImplicitActuator
    """Actuator class."""


@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    cls: type[ActuatorBase] = DCMotor
    """Actuator class."""

    saturation_effort: float = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    cls: type[ActuatorBase] = ActuatorNetLSTM

    network_file: str = MISSING
    """Path to the file containing network weights."""


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""

    cls: type[ActuatorBase] = ActuatorNetMLP

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
