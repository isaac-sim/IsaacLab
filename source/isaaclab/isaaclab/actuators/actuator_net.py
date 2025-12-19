# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural network models for actuators.

Currently, the following models are supported:

* Multi-Layer Perceptron (MLP)
* Long Short-Term Memory (LSTM)

"""

from isaaclab.utils.backend_utils import FactoryBase

from .actuator_base import ActuatorBase


class ActuatorNetLSTM(FactoryBase):
    """Factory for creating LSTM-based actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of an LSTM-based actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class ActuatorNetMLP(FactoryBase):
    """Factory for creating MLP-based actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of an MLP-based actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)
