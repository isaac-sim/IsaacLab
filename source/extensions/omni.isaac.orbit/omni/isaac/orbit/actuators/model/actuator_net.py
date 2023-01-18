# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Neural network models for actuators.

Currently, the following models are supported:
* Multi-Layer Perceptron (MLP)
* Long Short-Term Memory (LSTM)

"""

import torch
from typing import Sequence

from omni.isaac.orbit.utils.assets import read_file

from .actuator_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from .actuator_physics import DCMotor


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
        The class only supports desired joint positions commands as inputs in the method:
        :meth:`ActuatorNetMLP.set_command`.

    """

    cfg: ActuatorNetMLPCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetMLPCfg, num_actuators: int, num_envs: int, device: str):
        """Initializes the actuator net model.

        Args:
            cfg (ActuatorNetMLPCfg): The configuration for the actuator model.
            num_actuators (int): The number of actuators using the model.
            num_envs (int): The number of instances of the articulation.
            device (str): The computation device.
        """
        super().__init__(cfg, num_actuators, num_envs, device)
        # save config locally
        self.cfg = cfg
        # load the model from JIT file
        file_bytes = read_file(self.cfg.network_file)
        self.network = torch.jit.load(file_bytes).to(self.device)

        # create buffers for MLP history
        history_length = max(self.cfg.input_idx) + 1
        self._dof_pos_error_history = torch.zeros(self.num_envs, history_length, self.num_actuators, device=self.device)
        self._dof_vel_history = torch.zeros(self.num_envs, history_length, self.num_actuators, device=self.device)

    def reset(self, env_ids: Sequence[int]):
        self._dof_pos_error_history[env_ids] = 0.0
        self._dof_vel_history[env_ids] = 0.0

    def compute_torque(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        # move history queue by 1 and update top of history
        # -- positions
        self._dof_pos_error_history = self._dof_pos_error_history.roll(1, 1)
        self._dof_pos_error_history[:, 0] = self._des_dof_pos - dof_pos
        # -- velocity
        self._dof_vel_history = self._dof_vel_history.roll(1, 1)
        self._dof_vel_history[:, 0] = dof_vel

        # compute network inputs
        # -- positions
        pos_input = torch.cat([self._dof_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        pos_input = pos_input.reshape(self.num_envs * self.num_actuators, -1)
        # -- velocity
        vel_input = torch.cat([self._dof_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        vel_input = vel_input.reshape(self.num_envs * self.num_actuators, -1)
        # -- scale and concatenate inputs
        network_input = torch.cat([vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1)

        # run network inference
        desired_torques = self.network(network_input).reshape(self.num_envs, self.num_actuators)
        desired_torques = self.cfg.torque_scale * desired_torques
        # return torques
        return desired_torques


class ActuatorNetLSTM(DCMotor):
    """Actuator model based on recurrent neural network (LSTM).

    Unlike the MLP implementation :cite:t:`hwangbo2019learning`, this class implements
    the learned model as a temporal neural network (LSTM) based on the work from
    :cite:t:`rudin2022learning`. This removes the need of storing a history as the
    hidden states of the recurrent network captures the history.

    Note:
        The class only supports desired joint positions commands as inputs in the method:
        :meth:`ActuatorNetLSTM.set_command`.

    """

    cfg: ActuatorNetLSTMCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetLSTMCfg, num_actuators: int, num_envs: int, device: str):
        """Initializes the actuator net model.

        Args:
            cfg (ActuatorNetLSTMCfg): The configuration for the actuator model.
            num_actuators (int): The number of actuators using the model.
            num_envs (int): The number of instances of the articulation.
            device (str): The computation device.
        """
        super().__init__(cfg, num_actuators, num_envs, device)
        # load the model from JIT file
        file_bytes = read_file(self.cfg.network_file)
        self.network = torch.jit.load(file_bytes).to(self.device)

        # extract number of lstm layers and hidden dim from the shape of weights
        num_layers = len(self.network.lstm.state_dict()) // 4
        hidden_dim = self.network.lstm.state_dict()["weight_hh_l0"].shape[1]
        # create buffers for storing LSTM inputs
        self.sea_input = torch.zeros(self.num_envs * self.num_actuators, 1, 2, device=self.device)
        self.sea_hidden_state = torch.zeros(
            num_layers, self.num_envs * self.num_actuators, hidden_dim, device=self.device
        )
        self.sea_cell_state = torch.zeros(
            num_layers, self.num_envs * self.num_actuators, hidden_dim, device=self.device
        )
        # reshape via views (doesn't change the actual memory layout)
        layer_shape_per_env = (num_layers, self.num_envs, self.num_actuators, hidden_dim)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(layer_shape_per_env)
        self.sea_cell_state_per_env = self.sea_cell_state.view(layer_shape_per_env)

    def reset(self, env_ids: Sequence[int]):
        with torch.no_grad():
            self.sea_hidden_state_per_env[:, env_ids] = 0.0
            self.sea_cell_state_per_env[:, env_ids] = 0.0

    def compute_torque(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        # compute network inputs
        self.sea_input[:, 0, 0] = (self._des_dof_pos - dof_pos).flatten()
        self.sea_input[:, 0, 1] = dof_vel.flatten()
        # run network inference
        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )
        # return torques
        return torques.view(dof_pos.shape)
