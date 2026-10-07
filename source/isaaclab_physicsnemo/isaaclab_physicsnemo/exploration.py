# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploratory action sources for dynamics data collection."""

from __future__ import annotations

import math

import torch


class SmoothRandomActions:
    """Temporally correlated Gaussian actions for exploring robot dynamics.

    Each environment follows an independent first-order autoregressive (Ornstein-Uhlenbeck) process
    whose stationary distribution is zero-mean Gaussian with standard deviation :attr:`scale`. Smooth
    actions excite the joints over a wider frequency band than white noise without saturating the
    actuators every step. Finished environments restart from a fresh stationary sample.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        scale: float = 1.0,
        correlation_steps: float = 10.0,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize the action process.

        Args:
            num_envs: Number of environments.
            action_dim: Number of action channels per environment.
            scale: Stationary standard deviation of each action channel, in action units.
            correlation_steps: Number of steps over which the autocorrelation decays to ``1/e``.
            device: Device of the generated actions.
            seed: Seed of the process.

        Raises:
            ValueError: If ``scale`` or ``correlation_steps`` is not positive.
        """
        if scale <= 0.0 or correlation_steps <= 0.0:
            raise ValueError("scale and correlation_steps must be positive")
        self.scale = scale
        """Stationary standard deviation of each action channel."""
        self._decay = math.exp(-1.0 / correlation_steps)
        self._generator = torch.Generator(device=device).manual_seed(seed)
        self._actions = self._sample((num_envs, action_dim), device)

    def __call__(self, dones: torch.Tensor | None = None) -> torch.Tensor:
        """Advance the process by one step.

        Args:
            dones: Environments that finished in the previous step, shape [num_envs]. They restart from
                a fresh stationary sample.

        Returns:
            Actions for the next step, shape [num_envs, action_dim]. The tensor is owned by the caller.
        """
        innovation = self._sample(self._actions.shape, self._actions.device)
        self._actions = self._decay * self._actions + math.sqrt(1.0 - self._decay**2) * innovation
        if dones is not None:
            self._actions = torch.where(
                dones.unsqueeze(-1), self._sample(self._actions.shape, dones.device), self._actions
            )
        return self._actions.clone()

    def _sample(self, shape: torch.Size | tuple[int, ...], device: str | torch.device) -> torch.Tensor:
        return self.scale * torch.randn(shape, generator=self._generator, device=device)
