# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

from .actuator_pd_cfg import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg

if TYPE_CHECKING:
    from .actuator_net import ActuatorNetGRU, ActuatorNetGRUResidual, ActuatorNetLSTM, ActuatorNetMLP


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""

    class_type: type["ActuatorNetLSTM"] | str = "{DIR}.actuator_net:ActuatorNetLSTM"
    # we don't use stiffness and damping for actuator net
    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""

    class_type: type["ActuatorNetMLP"] | str = "{DIR}.actuator_net:ActuatorNetMLP"
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


@configclass
class ActuatorNetGRUCfg(IdealPDActuatorCfg):
    """Configuration for explicit full-torque GRU actuator models.

    This configures the :class:`~isaaclab.actuators.ActuatorNetGRU` model, where a recurrent
    (GRU) network predicts the *total* joint effort [N·m or N, depending on joint type]. The
    network is loaded as a TorchScript module from :attr:`network_file`. Since the network
    predicts the total effort directly, no PD gains are used; the computed effort is clipped to
    the actuator's effort limit by :meth:`~isaaclab.actuators.ActuatorBase._clip_effort`.
    """

    class_type: type["ActuatorNetGRU"] | str = "{DIR}.actuator_net:ActuatorNetGRU"
    # we don't use stiffness and damping since the network predicts the total effort
    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the TorchScript file containing the network weights.

    The loaded module must expose a ``.gru`` submodule (used to introspect the hidden and layer
    dimensions) and implement ``forward(x, hidden) -> (output, hidden)``, where ``x`` has shape
    (batch, 1, 3) carrying the joint position, position error, and velocity, ``hidden`` has shape
    (num_layers, batch, hidden_dim), and ``batch = num_envs * num_joints``. The ``output`` reshapes
    to (num_envs, num_joints).
    """

    position_normalization: tuple[float, float] | None = None
    """``(mean, std)`` applied to the joint position input as ``(x - mean) / std``.

    ``None`` (the default) disables normalization (identity).
    """

    pos_error_normalization: tuple[float, float] | None = None
    """``(mean, std)`` applied to the joint position error input as ``(x - mean) / std``.

    ``None`` (the default) disables normalization (identity).
    """

    vel_normalization: tuple[float, float] | None = None
    """``(mean, std)`` applied to the joint velocity input as ``(x - mean) / std``.

    ``None`` (the default) disables normalization (identity).
    """

    output_normalization: tuple[float, float] | None = None
    """Output denormalization as ``(mean, std)``.

    The raw network output ``y`` is denormalized as ``y * std + mean`` to recover the effort
    [N·m or N, depending on joint type]. ``None`` (the default) disables denormalization (identity).
    """


@configclass
class ActuatorNetGRUResidualCfg(ImplicitActuatorCfg):
    """Configuration for implicit-PD actuators with an added GRU residual effort.

    This configures the :class:`~isaaclab.actuators.ActuatorNetGRUResidual` model, an
    implicit-PD actuator whose feed-forward effort term is augmented by a recurrent (GRU)
    network predicting a *residual* effort [N·m or N, depending on joint type]. The PD term is
    handled by the physics engine using the configured :attr:`stiffness` and :attr:`damping`,
    while the network output is injected as the feed-forward effort.
    """

    class_type: type["ActuatorNetGRUResidual"] | str = "{DIR}.actuator_net:ActuatorNetGRUResidual"

    network_file: str = MISSING
    """Path to the TorchScript file containing the network weights.

    The loaded module must expose a ``.gru`` submodule (used to introspect the hidden and layer
    dimensions) and implement ``forward(x, hidden) -> (output, hidden)``, where ``x`` has shape
    (batch, 1, 3) carrying the joint position, position error, and velocity, ``hidden`` has shape
    (num_layers, batch, hidden_dim), and ``batch = num_envs * num_joints``. The ``output`` reshapes
    to (num_envs, num_joints).
    """

    position_normalization: tuple[float, float] | None = None
    """``(mean, std)`` applied to the joint position input as ``(x - mean) / std``.

    ``None`` (the default) disables normalization (identity).
    """

    pos_error_normalization: tuple[float, float] | None = None
    """``(mean, std)`` applied to the joint position error input as ``(x - mean) / std``.

    ``None`` (the default) disables normalization (identity).
    """

    vel_normalization: tuple[float, float] | None = None
    """``(mean, std)`` applied to the joint velocity input as ``(x - mean) / std``.

    ``None`` (the default) disables normalization (identity).
    """

    output_normalization: tuple[float, float] | None = None
    """Residual denormalization as ``(mean, std)``.

    The raw network output ``y`` is denormalized as ``y * std + mean`` to recover the residual
    effort [N·m or N, depending on joint type]. ``None`` (the default) disables denormalization
    (identity).
    """
