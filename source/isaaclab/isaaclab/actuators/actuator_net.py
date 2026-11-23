# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural network models for actuators.

Currently, the following models are supported:

* Multi-Layer Perceptron (MLP)
* Long Short-Term Memory (LSTM)

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from warp_nn.runtime import OnnxRuntime

from ..utils import index_fill_
from ..utils.assets import read_file
from ..utils.io import load_onnx_model
from ..utils.types import ArticulationActions
from ..utils.warp.memory import make_masked_fill
from .actuator_kernels import dc_motor_clip
from .actuator_pd import DCMotor

if TYPE_CHECKING:
    from .actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg


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
        if cfg.network_file is None:
            raise ValueError("network_file must be provided for the Isaac Lab neural actuator model.")
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

    def reset(self, env_ids: Sequence[int] | slice | None = None, env_mask: wp.array | None = None) -> None:
        if env_mask is not None:
            for state in (self.sea_hidden_state_per_env, self.sea_cell_state_per_env):
                values = wp.from_torch(state.flatten(2))
                wp.launch(
                    make_masked_fill(values.dtype, 3),
                    dim=values.shape,
                    inputs=[0.0, None, env_mask, values],
                    device=self._device,
                )
            return
        # reset the hidden and cell states for the specified environments
        with torch.no_grad():
            index_fill_(self.sea_hidden_state_per_env, env_ids, 0.0, dim=1)
            index_fill_(self.sea_cell_state_per_env, env_ids, 0.0, dim=1)

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
        self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)

        # return torques
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@wp.func
def _push_newest(history: wp.array3d(dtype=wp.float32), env_id: int, joint: int, sample: wp.float32):
    """Move the history of one (env, joint) back by one step and write ``sample`` as its newest entry."""
    for t in range(history.shape[1] - 1):
        k = history.shape[1] - 1 - t
        history[env_id, k, joint] = history[env_id, k - 1, joint]
    history[env_id, 0, joint] = sample


@wp.kernel(enable_backward=False)
def _update_mlp_history(
    # input
    joint_pos_target: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    history_indices: wp.array(dtype=wp.int32),
    pos_scale: wp.float32,
    vel_scale: wp.float32,
    pos_first: bool,
    # output
    joint_pos_error_history: wp.array3d(dtype=wp.float32),
    joint_vel_history: wp.array3d(dtype=wp.float32),
    network_input: wp.array2d(dtype=wp.float32),
):
    """Record the newest joint position error and velocity, then write the scaled samples the network reads.

    Each row of ``network_input`` belongs to one (env, joint): its position errors, then its velocities when
    ``pos_first``, otherwise the reverse. Launch with ``dim=(num_envs, num_joints)``.
    """
    env_id, joint = wp.tid()
    _push_newest(joint_pos_error_history, env_id, joint, joint_pos_target[env_id, joint] - joint_pos[env_id, joint])
    _push_newest(joint_vel_history, env_id, joint, joint_vel[env_id, joint])
    num_samples = history_indices.shape[0]
    row = env_id * joint_pos.shape[1] + joint
    pos_column = wp.where(pos_first, 0, num_samples)
    vel_column = num_samples - pos_column
    for i in range(num_samples):
        index = history_indices[i]
        network_input[row, pos_column + i] = pos_scale * joint_pos_error_history[env_id, index, joint]
        network_input[row, vel_column + i] = vel_scale * joint_vel_history[env_id, index, joint]


@wp.kernel(enable_backward=False)
def _scale_and_clip_mlp_torque(
    # input
    torque: wp.array(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    torque_scale: wp.float32,
    saturation_effort: wp.array2d(dtype=wp.float32),
    vel_at_effort_lim: wp.array2d(dtype=wp.float32),
    velocity_limit: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    # output
    computed_effort: wp.array2d(dtype=wp.float32),
    applied_effort: wp.array2d(dtype=wp.float32),
):
    """Scale the network torque of each (env, joint) and clip it to the DC motor limits.

    ``torque`` holds one value per (env, joint), row-major. Launch with ``dim=(num_envs, num_joints)``.
    """
    env_id, joint = wp.tid()
    effort = torque[env_id * computed_effort.shape[1] + joint] * torque_scale
    computed_effort[env_id, joint] = effort
    applied_effort[env_id, joint] = dc_motor_clip(
        effort,
        joint_vel[env_id, joint],
        saturation_effort[env_id, joint],
        vel_at_effort_lim[env_id, joint],
        velocity_limit[env_id, joint],
        effort_limit[env_id, joint],
    )


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
    from an ONNX file, or from a TorchScript file exported to ONNX, and runs with Warp-NN.

    Note:
        Only the desired joint positions are used as inputs to the network.

    """

    cfg: ActuatorNetMLPCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetMLPCfg, *args, **kwargs):
        if cfg.network_file is None:
            raise ValueError("network_file must be provided for the Isaac Lab neural actuator model.")
        super().__init__(cfg, *args, **kwargs)

        history_length = max(self.cfg.input_idx) + 1
        if min(self.cfg.input_idx) < -history_length:
            raise IndexError(f"MLP actuator history indices must be at least {-history_length}.")
        if self.cfg.input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )
        # the history samples the network reads; negative indices count back from the oldest sample
        indices = [index % history_length for index in self.cfg.input_idx]

        # load the network as an ONNX model and run it with Warp-NN, so that a CUDA graph can record the inference
        model = load_onnx_model(
            self.cfg.network_file,
            (torch.zeros(1, 2 * len(indices), device=self._device),),
            device=self._device,
            input_names=["input"],
            output_names=["torque"],
            dynamic_axes={"input": {0: "batch"}, "torque": {0: "batch"}},
        )
        self.network = OnnxRuntime(model, device=self._device)
        # an ONNX file keeps the input and output names it was exported with
        self._network_input_name = self.network.inputs[0].name
        self._network_output_name = self.network.outputs[0].name

        # create buffers for MLP history, newest sample first
        history_shape = (self._num_envs, history_length, self.num_joints)
        self._joint_pos_error_history = wp.zeros(history_shape, dtype=wp.float32, device=self._device)
        self._joint_vel_history = wp.zeros(history_shape, dtype=wp.float32, device=self._device)
        self._history_indices = wp.array(indices, dtype=wp.int32, device=self._device)
        # the network input, one row per (env, joint), and the network outputs it allocates once
        input_shape = (self._num_envs * self.num_joints, 2 * len(indices))
        self._network_input = wp.zeros(input_shape, dtype=wp.float32, device=self._device)
        self.network.prepare({self._network_input_name: self._network_input})
        # the efforts are written by a Warp kernel; the actuator's effort attributes are Torch views of them
        effort_shape = (self._num_envs, self.num_joints)
        self._computed_effort = wp.zeros(effort_shape, dtype=wp.float32, device=self._device)
        self._applied_effort = wp.zeros(effort_shape, dtype=wp.float32, device=self._device)
        self.computed_effort = wp.to_torch(self._computed_effort)
        self.applied_effort = wp.to_torch(self._applied_effort)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None, env_mask: wp.array | None = None) -> None:
        # reset the history for the specified environments
        for history in (self._joint_pos_error_history, self._joint_vel_history):
            if env_mask is not None:
                wp.launch(
                    make_masked_fill(history.dtype, 3),
                    dim=history.shape,
                    inputs=[0.0, env_mask, None, history],
                    device=self._device,
                )
            else:
                index_fill_(wp.to_torch(history), env_ids, 0.0)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # move history queue by 1, update top of history and write the scaled network input
        wp.launch(
            _update_mlp_history,
            dim=(self._num_envs, self.num_joints),
            inputs=[
                wp.from_torch(control_action.joint_positions),
                wp.from_torch(joint_pos),
                wp.from_torch(joint_vel),
                self._history_indices,
                self.cfg.pos_scale,
                self.cfg.vel_scale,
                self.cfg.input_order == "pos_vel",
            ],
            outputs=[self._joint_pos_error_history, self._joint_vel_history, self._network_input],
            device=self._device,
        )
        # run network inference
        torque = self.network({self._network_input_name: self._network_input})[self._network_output_name]
        # scale the torque and clip it based on the motor limits
        wp.launch(
            _scale_and_clip_mlp_torque,
            dim=(self._num_envs, self.num_joints),
            inputs=[
                torque.reshape((self._num_envs * self.num_joints,)),
                wp.from_torch(joint_vel),
                self.cfg.torque_scale,
                wp.from_torch(self._saturation_effort),
                wp.from_torch(self._vel_at_effort_lim),
                wp.from_torch(self.actuator_velocity_limit),
                wp.from_torch(self.actuator_effort_limit),
            ],
            outputs=[self._computed_effort, self._applied_effort],
            device=self._device,
        )

        # return torques
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
