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
    whose stationary distribution is zero-mean Gaussian with the environment's standard deviation in
    :attr:`scales`. Smooth actions excite the joints over a wider frequency band than white noise without
    saturating the actuators every step. Finished environments restart from a fresh stationary sample.

    A scale range gives every episode its own scale, drawn uniformly from the range. Data recorded this
    way covers both gentle and saturating commands, so a learned model stays accurate for controllers
    that use the full actuator range.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        scale: float | tuple[float, float] = 1.0,
        correlation_steps: float = 10.0,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize the action process.

        Args:
            num_envs: Number of environments.
            action_dim: Number of action channels per environment.
            scale: Stationary standard deviation of each action channel, in action units, or a
                ``(low, high)`` range from which each episode draws its standard deviation.
            correlation_steps: Number of steps over which the autocorrelation decays to ``1/e``.
            device: Device of the generated actions.
            seed: Seed of the process.

        Raises:
            ValueError: If a scale or ``correlation_steps`` is not positive, or the scale range is empty.
        """
        low, high = (scale, scale) if isinstance(scale, (int, float)) else scale
        if low <= 0.0 or high < low or correlation_steps <= 0.0:
            raise ValueError("scales must form a positive range and correlation_steps must be positive")
        self._scale_range = (float(low), float(high))
        self._decay = math.exp(-1.0 / correlation_steps)
        self._generator = torch.Generator(device=device).manual_seed(seed)
        self._noise = torch.randn((num_envs, action_dim), generator=self._generator, device=device)
        self.scales = self._sample_scales(num_envs, device)
        """Stationary standard deviation of each environment's current episode, shape [num_envs, 1]."""

    def __call__(self, dones: torch.Tensor | None = None) -> torch.Tensor:
        """Advance the process by one step.

        Args:
            dones: Environments that finished in the previous step, shape [num_envs]. They restart from
                a fresh stationary sample with a newly drawn scale.

        Returns:
            Actions for the next step, shape [num_envs, action_dim]. The tensor is owned by the caller.
        """
        innovation = torch.randn(self._noise.shape, generator=self._generator, device=self._noise.device)
        self._noise = self._decay * self._noise + math.sqrt(1.0 - self._decay**2) * innovation
        if dones is not None:
            restart = dones.unsqueeze(-1)
            fresh = torch.randn(self._noise.shape, generator=self._generator, device=self._noise.device)
            self._noise = torch.where(restart, fresh, self._noise)
            self.scales = torch.where(restart, self._sample_scales(len(dones), dones.device), self.scales)
        return self.scales * self._noise

    def _sample_scales(self, num_envs: int, device: str | torch.device) -> torch.Tensor:
        low, high = self._scale_range
        uniform = torch.rand((num_envs, 1), generator=self._generator, device=device)
        return low + (high - low) * uniform


class UniformRandomActions:
    """Independent uniform actions, the joint-target protocol of FlashNeRD data.

    Unlike :class:`SmoothRandomActions`, consecutive targets are uncorrelated, so a position-controlled robot
    sees step changes of its targets over the whole range. Each target is held for ``hold_steps``
    environment steps, so recording at the physics rate still changes targets at the control rate. Finished
    environments draw a fresh target immediately.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        high: float,
        hold_steps: int = 1,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize the action source.

        Args:
            num_envs: Number of environments.
            action_dim: Number of action channels per environment.
            high: Bound of the symmetric uniform range ``[-high, high]``, in action units.
            hold_steps: Number of consecutive steps that keep one target.
            device: Device of the generated actions.
            seed: Seed of the source.

        Raises:
            ValueError: If ``high`` or ``hold_steps`` is not positive.
        """
        if high <= 0.0 or hold_steps < 1:
            raise ValueError("high and hold_steps must be positive")
        self.high = high
        """Bound of the symmetric uniform action range."""
        self.hold_steps = hold_steps
        """Number of consecutive steps that keep one target."""
        self._generator = torch.Generator(device=device).manual_seed(seed)
        self._actions = self._sample((num_envs, action_dim), device)
        self._step = 0

    def __call__(self, dones: torch.Tensor | None = None) -> torch.Tensor:
        """Return the actions of the next step.

        Args:
            dones: Environments that finished in the previous step, shape [num_envs]. They draw a fresh target.

        Returns:
            Actions for the next step, shape [num_envs, action_dim]. The tensor is owned by the caller.
        """
        if self._step > 0 and self._step % self.hold_steps == 0:
            self._actions = self._sample(self._actions.shape, self._actions.device)
        elif dones is not None:
            fresh = self._sample(self._actions.shape, self._actions.device)
            self._actions = torch.where(dones.unsqueeze(-1), fresh, self._actions)
        self._step += 1
        return self._actions.clone()

    def _sample(self, shape: torch.Size | tuple[int, ...], device: str | torch.device) -> torch.Tensor:
        uniform = torch.rand(shape, generator=self._generator, device=device)
        return self.high * (2.0 * uniform - 1.0)
