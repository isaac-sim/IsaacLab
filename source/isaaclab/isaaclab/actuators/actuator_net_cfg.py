# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from . import actuator_net
from .actuator_pd_cfg import DCMotorCfg


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    class_type: type = actuator_net.ActuatorNetLSTM
    # we don't use stiffness and damping for actuator net
    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""

    class_type: type = actuator_net.ActuatorNetMLP
    # we don't use stiffness and damping for actuator net

    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""

    pos_scale: float = MISSING
    """Scaling of the joint position errors input to the network."""
    vel_scale: float = MISSING
    """Scaling of the joint velocities input to the network."""
    torque_scale: float = MISSING
    """Scaling of the joint efforts output from the network."""

    input_order: Literal["pos_vel", "vel_pos"] = MISSING
    """Order of the inputs to the network.

    The order can be one of the following:

    * ``"pos_vel"``: joint position errors followed by joint velocities
    * ``"vel_pos"``: joint velocities followed by joint position errors
    """

    input_idx: Iterable[int] = MISSING
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """
