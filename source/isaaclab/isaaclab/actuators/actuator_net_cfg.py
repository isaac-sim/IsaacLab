# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from isaaclab.utils import REQUIRED

from .actuator_pd_cfg import DCMotorCfg

if TYPE_CHECKING:
    from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP


@dataclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    class_type: type["ActuatorNetLSTM"] | str = "isaaclab.actuators.actuator_net:ActuatorNetLSTM"
    # we don't use stiffness and damping for actuator net
    stiffness: Any = None
    damping: Any = None

    network_file: str = REQUIRED
    """Path to the file containing network weights."""


@dataclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""

    class_type: type["ActuatorNetMLP"] | str = "isaaclab.actuators.actuator_net:ActuatorNetMLP"
    # we don't use stiffness and damping for actuator net

    stiffness: Any = None
    damping: Any = None

    network_file: str = REQUIRED
    """Path to the file containing network weights."""

    pos_scale: float = REQUIRED
    """Scaling of the joint position errors input to the network."""
    vel_scale: float = REQUIRED
    """Scaling of the joint velocities input to the network."""
    torque_scale: float = REQUIRED
    """Scaling of the joint efforts output from the network."""

    input_order: Literal["pos_vel", "vel_pos"] = REQUIRED
    """Order of the inputs to the network.

    The order can be one of the following:

    * ``"pos_vel"``: joint position errors followed by joint velocities
    * ``"vel_pos"``: joint velocities followed by joint position errors
    """

    input_idx: Iterable[int] = REQUIRED
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """
