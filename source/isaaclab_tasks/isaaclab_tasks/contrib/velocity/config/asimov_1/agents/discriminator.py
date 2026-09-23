# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation
from torch import autograd


class AMPFeatureNormalizer(nn.Module):
    """Running feature normalizer for expert and policy motion states."""

    def __init__(self, observation_dim: int, epsilon: float = 1.0e-4, clip_obs: float = 10.0) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.register_buffer("_mean", torch.zeros(1, observation_dim, dtype=torch.float64))
        self.register_buffer("_var", torch.ones(1, observation_dim, dtype=torch.float64))
        self.register_buffer("_std", torch.ones(1, observation_dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1.0e-4, dtype=torch.float64))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(device=values.device, dtype=values.dtype)
        variance = self._var.to(device=values.device, dtype=values.dtype)
        normalized = (values - mean) / torch.sqrt(variance + self.epsilon)
        return torch.clamp(normalized, -self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        array = values.detach().cpu().numpy()
        batch_mean = np.mean(array, axis=0, keepdims=True)
        batch_var = np.var(array, axis=0, keepdims=True)
        batch_count = array.shape[0]

        mean = self._mean.cpu().numpy()
        variance = self._var.cpu().numpy()
        count = float(self.count.item())
        delta = batch_mean - mean
        total_count = count + batch_count
        new_mean = mean + delta * batch_count / total_count
        moment_a = variance * count
        moment_b = batch_var * batch_count
        moment_2 = moment_a + moment_b + np.square(delta) * count * batch_count / total_count
        new_variance = moment_2 / total_count

        self._mean.copy_(torch.as_tensor(new_mean, dtype=self._mean.dtype, device=self._mean.device))
        self._var.copy_(torch.as_tensor(new_variance, dtype=self._var.dtype, device=self._var.device))
        self._std.copy_(torch.sqrt(self._var))
        self.count.fill_(total_count)


class AMPDiscriminator(nn.Module):
    """Least-squares discriminator over consecutive motion states."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256),
        activation: str = "relu",
        feature_normalization: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.observation_dim = observation_dim
        self.input_dim = 2 * observation_dim

        layers = []
        current_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(resolve_nn_activation(activation))
            current_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(current_dim, 1)

        self.feature_normalization = feature_normalization
        if feature_normalization:
            self.feature_norm = AMPFeatureNormalizer(observation_dim)

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        return self.linear(self.trunk(transition))

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        if self.feature_normalization:
            return self.feature_norm(state)
        return state

    @torch.no_grad()
    def predict_amp_reward(self, state: torch.Tensor, next_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        normalized_state = self.normalize(state)
        normalized_next_state = self.normalize(next_state)
        prediction = self(torch.cat((normalized_state, normalized_next_state), dim=-1))
        reward = torch.clamp(1.0 - 0.25 * torch.square(prediction - 1.0), min=0.0)
        self.train()
        return reward.squeeze(-1), prediction.squeeze(-1)

    def compute_grad_pen(
        self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
    ) -> torch.Tensor:
        expert_transition = torch.cat((expert_state, expert_next_state), dim=-1).detach().requires_grad_(True)
        prediction = self(expert_transition)
        gradient = autograd.grad(
            outputs=prediction,
            inputs=expert_transition,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * gradient.norm(2, dim=1).pow(2).mean()

    @torch.no_grad()
    def update_normalization(self, policy_state: torch.Tensor, expert_state: torch.Tensor) -> None:
        if not self.feature_normalization:
            return
        normalized_policy_state = self.normalize(policy_state)
        normalized_expert_state = self.normalize(expert_state)
        self.feature_norm.update(normalized_policy_state)
        self.feature_norm.update(normalized_expert_state)
