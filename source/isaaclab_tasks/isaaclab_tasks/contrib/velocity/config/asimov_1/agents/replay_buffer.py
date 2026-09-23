# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class AMPReplayBuffer:
    """Fixed-capacity ring buffer of AMP state transitions."""

    def __init__(self, observation_dim: int, capacity: int, device: str) -> None:
        if capacity <= 0:
            raise ValueError(f"AMP replay-buffer capacity must be positive, got {capacity}.")

        self.states = torch.empty(capacity, observation_dim, device=device)
        self.next_states = torch.empty(capacity, observation_dim, device=device)
        self.capacity = capacity
        self._cursor = 0
        self.num_samples = 0

    @torch.no_grad()
    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        states = states.detach()
        next_states = next_states.detach()
        if states.shape != next_states.shape:
            raise ValueError(
                f"AMP states and next states must have the same shape, got {states.shape} and {next_states.shape}."
            )
        if states.shape[-1] != self.states.shape[-1]:
            raise ValueError(f"Expected AMP states with {self.states.shape[-1]} features, got {states.shape[-1]}.")

        if len(states) >= self.capacity:
            self.states.copy_(states[-self.capacity :])
            self.next_states.copy_(next_states[-self.capacity :])
            self._cursor = 0
            self.num_samples = self.capacity
            return

        end = self._cursor + len(states)
        if end <= self.capacity:
            self.states[self._cursor : end].copy_(states)
            self.next_states[self._cursor : end].copy_(next_states)
        else:
            first_count = self.capacity - self._cursor
            self.states[self._cursor :].copy_(states[:first_count])
            self.next_states[self._cursor :].copy_(next_states[:first_count])
            self.states[: end - self.capacity].copy_(states[first_count:])
            self.next_states[: end - self.capacity].copy_(next_states[first_count:])

        self._cursor = end % self.capacity
        self.num_samples = min(self.capacity, self.num_samples + len(states))

    def generator(self, num_batches: int, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty AMP replay buffer.")
        for _ in range(num_batches):
            indices = np.random.choice(self.num_samples, size=batch_size)
            yield self.states[indices], self.next_states[indices]
