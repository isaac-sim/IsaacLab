# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural network models for actuators.

Currently, the following models are supported:

* Multi-Layer Perceptron (MLP)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU), both explicit full-torque and implicit-PD residual variants

"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.assets import read_file
from isaaclab.utils.types import ArticulationActions

from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator

if TYPE_CHECKING:
    from .actuator_net_cfg import (
        ActuatorNetGRUCfg,
        ActuatorNetGRUResidualCfg,
        ActuatorNetLSTMCfg,
        ActuatorNetMLPCfg,
    )

logger = logging.getLogger(__name__)


class ActuatorNetLSTM(DCMotor):
    """Actuator model based on recurrent neural network (LSTM).

    Unlike the MLP implementation :cite:t:`hwangbo2019learning`, this class implements
    the learned model as a temporal neural network (LSTM) based on the work from
    :cite:t:`rudin2022learning`. This removes the need of storing a history as the
    hidden states of the recurrent network captures the history.

    Note:
        Only the desired joint positions are used as inputs to the network.
    """

    cfg: ActuatorNetLSTMCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetLSTMCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # load the model from JIT file
        file_bytes = read_file(self.cfg.network_file)
        self.network = torch.jit.load(file_bytes, map_location=self._device).eval()

        # extract number of lstm layers and hidden dim from the shape of weights
        num_layers = len(self.network.lstm.state_dict()) // 4
        hidden_dim = self.network.lstm.state_dict()["weight_hh_l0"].shape[1]
        # create buffers for storing LSTM inputs
        self.sea_input = torch.zeros(self._num_envs * self.num_joints, 1, 2, device=self._device)
        self.sea_hidden_state = torch.zeros(
            num_layers, self._num_envs * self.num_joints, hidden_dim, device=self._device
        )
        self.sea_cell_state = torch.zeros(num_layers, self._num_envs * self.num_joints, hidden_dim, device=self._device)
        # reshape via views (doesn't change the actual memory layout)
        layer_shape_per_env = (num_layers, self._num_envs, self.num_joints, hidden_dim)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(layer_shape_per_env)
        self.sea_cell_state_per_env = self.sea_cell_state.view(layer_shape_per_env)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        # reset the hidden and cell states for the specified environments
        with torch.no_grad():
            self.sea_hidden_state_per_env[:, env_ids] = 0.0
            self.sea_cell_state_per_env[:, env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute network inputs
        self.sea_input[:, 0, 0] = (control_action.joint_positions - joint_pos).flatten()
        self.sea_input[:, 0, 1] = joint_vel.flatten()

        # run network inference
        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )
        self.computed_effort = torques.reshape(self._num_envs, self.num_joints)

        # clip the computed effort based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # return torques
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class _GRUActuatorMixin:
    """Shared machinery for the GRU-based actuator models.

    Loads the TorchScript GRU network, allocates the recurrent input and hidden-state buffers, and
    runs inference. The network consumes a fixed input of joint position, position error, and
    velocity. An optional ``(mean, std)`` normalization may be applied to each input and to the
    output (``None`` selects the identity transform). The concrete actuator classes combine this
    mixin with an explicit (:class:`IdealPDActuator`) or implicit (:class:`ImplicitActuator`) base
    to define their effort semantics.
    """

    # number of fixed network inputs: [position, position_error, velocity]
    _NUM_INPUTS = 3
    # standard-deviation floor used when normalizing to avoid division by tiny values
    _GRU_STD_FLOOR = 1.0e-8

    def _init_gru_runtime(self) -> None:
        """Load the network and allocate the GRU buffers and normalization statistics.

        Raises:
            ValueError: If the TorchScript module does not expose a ``.gru`` submodule, or if its
                input dimension is not 3 (joint position, position error, and velocity).
        """
        # load the TorchScript network
        file_bytes = read_file(self.cfg.network_file)
        self.network = torch.jit.load(file_bytes, map_location=self._device).eval()
        if not hasattr(self.network, "gru"):
            raise ValueError(f"The network file '{self.cfg.network_file}' must expose a TorchScript '.gru' submodule.")

        # infer dimensions from the GRU weights (the input is [position, position_error, velocity])
        gru_state = self.network.gru.state_dict()
        if any("reverse" in key for key in gru_state):
            raise ValueError(
                f"The network file '{self.cfg.network_file}' uses a bidirectional GRU, which is not supported."
            )
        input_dim = int(gru_state["weight_ih_l0"].shape[1])
        hidden_dim = int(gru_state["weight_hh_l0"].shape[1])
        num_layers = sum(1 for key in gru_state if key.startswith("weight_ih_l") and "reverse" not in key)
        if input_dim != self._NUM_INPUTS:
            raise ValueError(
                f"The network file '{self.cfg.network_file}' must take {self._NUM_INPUTS} inputs (joint position,"
                f" position error, and velocity), but its GRU expects {input_dim}."
            )

        # resolve (mean, std) normalization for the inputs and output (identity when unset)
        self._position_norm = self._resolve_normalization(self.cfg.position_normalization, "position_normalization")
        self._pos_error_norm = self._resolve_normalization(self.cfg.pos_error_normalization, "pos_error_normalization")
        self._vel_norm = self._resolve_normalization(self.cfg.vel_normalization, "vel_normalization")
        self._output_norm = self._resolve_normalization(self.cfg.output_normalization, "output_normalization")

        # recurrent input and hidden-state buffers
        batch = self._num_envs * self.num_joints
        self.sea_input = torch.zeros(batch, 1, self._NUM_INPUTS, device=self._device)
        self.sea_hidden_state = torch.zeros(num_layers, batch, hidden_dim, device=self._device)
        # per-env view for resets (shares storage)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            num_layers, self._num_envs, self.num_joints, hidden_dim
        )

    def _resolve_normalization(self, stats: tuple[float, float] | None, name: str) -> tuple[float, float]:
        """Return the ``(mean, std)`` to apply, defaulting to identity and flooring the std.

        Args:
            stats: The ``(mean, std)`` pair, or None for the identity transform.
            name: The configuration field name, used for the warning message.

        Returns:
            The resolved ``(mean, std)`` with the std floored to avoid division by tiny values.
        """
        if stats is None:
            return 0.0, 1.0
        mean, std = float(stats[0]), float(stats[1])
        if std < 0.0:
            raise ValueError(
                f"Actuator '{self.cfg.network_file}' has {name} std={std}; the standard deviation must be"
                " non-negative. Check the (mean, std) ordering."
            )
        if std < self._GRU_STD_FLOOR:
            logger.warning(
                "Actuator '%s' has %s std=%s below the floor %s; flooring it, which can amplify the"
                " normalized values. Set a larger std or leave the field unset for identity.",
                self.cfg.network_file,
                name,
                std,
                self._GRU_STD_FLOOR,
            )
        return mean, max(std, self._GRU_STD_FLOOR)

    def _reset_gru_state(self, env_ids: Sequence[int]):
        """Zero the GRU hidden state for the specified environments.

        Args:
            env_ids: The environment indices whose hidden state should be reset.
        """
        with torch.no_grad():
            self.sea_hidden_state_per_env[:, env_ids] = 0.0

    def _predict_gru_effort(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> torch.Tensor:
        """Assemble the network input, run inference, and return the denormalized effort.

        Args:
            control_action: The joint action instance holding the desired joint positions.
            joint_pos: The current joint positions. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities. Shape is (num_envs, num_joints).

        Returns:
            The predicted effort [N·m or N, depending on joint type]. Shape is
            (num_envs, num_joints).

        Raises:
            ValueError: If ``control_action.joint_positions`` is None.
        """
        if control_action.joint_positions is None:
            raise ValueError("GRU actuator input requires control_action.joint_positions to be set.")
        # normalized [position, position_error, velocity] inputs
        position = joint_pos.flatten()
        pos_error = (control_action.joint_positions - joint_pos).flatten()
        velocity = joint_vel.flatten()
        self.sea_input[:, 0, 0] = (position - self._position_norm[0]) / self._position_norm[1]
        self.sea_input[:, 0, 1] = (pos_error - self._pos_error_norm[0]) / self._pos_error_norm[1]
        self.sea_input[:, 0, 2] = (velocity - self._vel_norm[0]) / self._vel_norm[1]

        # run inference, then denormalize and guard against a non-finite output
        with torch.inference_mode():
            output, self.sea_hidden_state[:] = self.network(self.sea_input, self.sea_hidden_state)
            output = output * self._output_norm[1] + self._output_norm[0]
            # a non-finite prediction carries no usable actuation, so command zero effort this step
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            return output.reshape(self._num_envs, self.num_joints)


class ActuatorNetGRU(_GRUActuatorMixin, IdealPDActuator):
    """Explicit actuator model based on a recurrent neural network (GRU).

    The GRU network predicts the *total* joint effort [N·m or N, depending on joint type] from the
    joint position, position error, and velocity. Unlike the analytical models, no PD gains are
    applied; the hidden state of the recurrent network captures the actuator history. The predicted
    effort is clipped to the actuator's effort limit via :meth:`~isaaclab.actuators.ActuatorBase._clip_effort`.

    This model derives from :class:`IdealPDActuator`, whose simple symmetric ``±effort_limit``
    saturation matches a learned total-torque source without requiring the velocity-dependent
    torque-speed parameters of a DC motor.

    Note:
        The recurrent hidden state encodes the actuator history and is only cleared by
        :meth:`reset`. Callers must reset the relevant environments on episode boundaries
        (and after any control gap, e.g. a hardware reconnect) so the first post-reset effort is
        not computed against stale temporal context.
    """

    cfg: ActuatorNetGRUCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetGRUCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._init_gru_runtime()

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._reset_gru_state(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self.computed_effort = self._predict_gru_effort(control_action, joint_pos, joint_vel)
        # clip the computed effort based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class ActuatorNetGRUResidual(_GRUActuatorMixin, ImplicitActuator):
    """Implicit-PD actuator model with an added recurrent (GRU) residual effort.

    This model behaves like an :class:`ImplicitActuator` -- the physics engine applies the PD
    control using the configured stiffness and damping -- but augments the feed-forward effort
    term with a *residual* effort [N·m or N, depending on joint type] predicted by a recurrent
    (GRU) network. The residual is added to any existing feed-forward effort, and the approximate
    total effort is stored for reward computation while the desired joint positions and velocities
    are preserved so the engine can compute the PD term.

    Note:
        As with any :class:`ImplicitActuator`, the effort actually applied by the engine is the
        feed-forward effort plus the engine-side PD term, and it is bounded by the simulation
        effort limit (``effort_limit_sim``) rather than by :meth:`~isaaclab.actuators.ActuatorBase._clip_effort`
        (which only populates the reported :attr:`applied_effort`). Set ``effort_limit_sim`` to a
        finite value to bound the residual feed-forward. The hidden state is cleared only by
        :meth:`reset`; reset the relevant environments on episode boundaries (and after any control
        gap) to avoid stale recurrent context.
    """

    cfg: ActuatorNetGRUResidualCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetGRUResidualCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._init_gru_runtime()

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._reset_gru_state(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # add the GRU residual to the feed-forward effort
        residual = self._predict_gru_effort(control_action, joint_pos, joint_vel)
        if control_action.joint_efforts is None:
            control_action.joint_efforts = residual
        else:
            control_action.joint_efforts = control_action.joint_efforts + residual

        # approximate total effort for reward telemetry (engine applies the PD term)
        error_pos = control_action.joint_positions - joint_pos
        if control_action.joint_velocities is not None:
            error_vel = control_action.joint_velocities - joint_vel
        else:
            error_vel = -joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.applied_effort = self._clip_effort(self.computed_effort)
        # positions/velocities are preserved so the engine computes the PD term
        return control_action


class ActuatorNetMLP(DCMotor):
    """Actuator model based on multi-layer perceptron and joint history.

    Many times the analytical model is not sufficient to capture the actuator dynamics, the
    delay in the actuator response, or the non-linearities in the actuator. In these cases,
    a neural network model can be used to approximate the actuator dynamics. This model is
    trained using data collected from the physical actuator and maps the joint state and the
    desired joint command to the produced torque by the actuator.

    This class implements the learned model as a neural network based on the work from
    :cite:t:`hwangbo2019learning`. The class stores the history of the joint positions errors
    and velocities which are used to provide input to the neural network. The model is loaded
    as a TorchScript.

    Note:
        Only the desired joint positions are used as inputs to the network.

    """

    cfg: ActuatorNetMLPCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetMLPCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # load the model from JIT file
        file_bytes = read_file(self.cfg.network_file)
        self.network = torch.jit.load(file_bytes, map_location=self._device).eval()

        # create buffers for MLP history
        history_length = max(self.cfg.input_idx) + 1
        self._joint_pos_error_history = torch.zeros(
            self._num_envs, history_length, self.num_joints, device=self._device
        )
        self._joint_vel_history = torch.zeros(self._num_envs, history_length, self.num_joints, device=self._device)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        # reset the history for the specified environments
        self._joint_pos_error_history[env_ids] = 0.0
        self._joint_vel_history[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # move history queue by 1 and update top of history
        # -- positions
        self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
        self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
        # -- velocity
        self._joint_vel_history = self._joint_vel_history.roll(1, 1)
        self._joint_vel_history[:, 0] = joint_vel
        # save current joint vel for dc-motor clipping
        self._joint_vel[:] = joint_vel

        # compute network inputs
        # -- positions
        pos_input = torch.cat([self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        pos_input = pos_input.view(self._num_envs * self.num_joints, -1)
        # -- velocity
        vel_input = torch.cat([self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        vel_input = vel_input.view(self._num_envs * self.num_joints, -1)
        # -- scale and concatenate inputs
        if self.cfg.input_order == "pos_vel":
            network_input = torch.cat([pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale], dim=1)
        elif self.cfg.input_order == "vel_pos":
            network_input = torch.cat([vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1)
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        # run network inference
        with torch.inference_mode():
            torques = self.network(network_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale

        # clip the computed effort based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # return torques
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
