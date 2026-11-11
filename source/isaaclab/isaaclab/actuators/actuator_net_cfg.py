# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from ..utils import configclass
from .actuator_pd_cfg import DCMotorCfg

if TYPE_CHECKING:
    from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    class_type: type["ActuatorNetLSTM"] | str = "{DIR}.actuator_net:ActuatorNetLSTM"
    # we don't use stiffness and damping for actuator net
    stiffness = None
    damping = None

    network_file: str | None = None
    """Path to the TorchScript checkpoint. Required by the Isaac Lab model and for a new native neural drive.

    With :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators`, None keeps an existing USD actuator's checkpoint.
    """


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model.

    Input settings are required when supplying :attr:`network_file`. Leaving the checkpoint and its
    input settings unset keeps an existing USD actuator's checkpoint and preprocessing metadata.
    """

    class_type: type["ActuatorNetMLP"] | str = "{DIR}.actuator_net:ActuatorNetMLP"
    # we don't use stiffness and damping for actuator net

    stiffness = None
    damping = None

    network_file: str | None = None
    """Path to the TorchScript checkpoint. Required by the Isaac Lab model and for a new native neural drive.

    With :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators`, None keeps an existing USD actuator's checkpoint.
    """

    pos_scale: float | None = None
    """Scaling of the joint position errors input to the network."""
    vel_scale: float | None = None
    """Scaling of the joint velocities input to the network."""
    torque_scale: float | None = None
    """Scaling of the joint efforts output from the network."""

    input_order: Literal["pos_vel", "vel_pos"] | None = None
    """Order of the inputs to the network.

    The order can be one of the following:

    * ``"pos_vel"``: joint position errors followed by joint velocities
    * ``"vel_pos"``: joint velocities followed by joint position errors
    """

    input_idx: Iterable[int] | None = None
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """

    def validate_config(self) -> None:
        """Require MLP preprocessing settings together with a replacement checkpoint.

        Raises:
            ValueError: If metadata is configured without a checkpoint, or a supplied checkpoint lacks input settings.
        """
        metadata = {
            "input_order": self.input_order,
            "input_idx": self.input_idx,
            "pos_scale": self.pos_scale,
            "vel_scale": self.vel_scale,
            "torque_scale": self.torque_scale,
        }
        if self.network_file is None:
            if any(value is not None for value in metadata.values()):
                raise ValueError("network_file is required when configuring MLP checkpoint metadata.")
            return
        missing = [name for name, value in metadata.items() if value is None]
        if missing:
            raise ValueError(f"network_file requires MLP metadata fields: {', '.join(missing)}.")
