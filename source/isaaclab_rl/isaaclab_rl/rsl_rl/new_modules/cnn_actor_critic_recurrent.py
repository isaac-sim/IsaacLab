# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation
from torch import nn


class CNNActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        cnn_kernel_size=3,
        cnn_stride=3,
        cnn_filters=[32, 16, 8],
        paddings=[0, 0, 0],
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        out_channels = cnn_filters
        in_channels = [1] + cnn_filters[:-1]

        actor_layers = []
        input_dim_a = num_actor_obs
        for in_ch, out_ch, pad in zip(in_channels, out_channels, paddings):
            actor_layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=cnn_kernel_size,
                stride=cnn_stride,
                padding=pad
            ))
            actor_layers.append(activation)
            input_dim_a = (input_dim_a + 2 * pad - cnn_kernel_size) // cnn_stride + 1

        actor_layers.append(nn.Flatten())
        input_dim_a = input_dim_a * out_channels[-1]
        self.cnn_a = nn.Sequential(*actor_layers)

        critic_layers = []
        input_dim_c = num_critic_obs
        for in_ch, out_ch, pad in zip(in_channels, out_channels, paddings):
            critic_layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=cnn_kernel_size,
                stride=cnn_stride,
                padding=pad
            ))
            critic_layers.append(activation)
            input_dim_c = (input_dim_c + 2 * pad - cnn_kernel_size) // cnn_stride + 1

        critic_layers.append(nn.Flatten())
        input_dim_c = input_dim_c * out_channels[-1]
        self.cnn_c = nn.Sequential(*critic_layers)

        self.memory_a = Memory(input_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(input_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.cnn_a, self.memory_a}")
        print(f"Critic RNN: {self.cnn_c, self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        if observations.dim() == 3:
            conv_obs = self.cnn_a(observations.reshape(-1, observations.shape[-1]).unsqueeze(1))
            conv_obs = conv_obs.squeeze(1).reshape(*observations.shape[:2], -1)
        else:
            conv_obs = self.cnn_a(observations.unsqueeze(1)).squeeze(1)

        input_a = self.memory_a(conv_obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        conv_obs = self.cnn_a(observations.unsqueeze(1)).squeeze(1)
        input_a = self.memory_a(conv_obs)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        if critic_observations.dim() == 3:
            conv_obs = self.cnn_c(critic_observations.reshape(-1, critic_observations.shape[-1]).unsqueeze(1))
            conv_obs = conv_obs.squeeze(1).reshape(*critic_observations.shape[:2], -1)
        else:
            conv_obs = self.cnn_c(critic_observations.unsqueeze(1)).squeeze(1)
        input_c = self.memory_c(conv_obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states