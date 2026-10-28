# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode statistics of each Policy Ghost slot."""

from __future__ import annotations

from collections import deque

import torch

from isaaclab.visualizers.policy_ghost import PolicyGhostMetrics

_WINDOW = 20
"""Most recent finished episodes each slot's statistics average over."""


class PolicyGhostMetricsTracker:
    """Tracks the return and length of the episodes each slot's environments finish.

    Slots run in lockstep with the reference: a ghost whose episode ends early waits for the reference to reset, so its
    episode is counted once, when it first terminates, and a ghost that is still running when the reference resets
    is counted then.
    """

    def __init__(self, slice_size: int, num_slots: int, num_envs: int, step_dt: float, device: torch.device | str):
        """Create the tracker.

        Args:
            slice_size: Environments per slot.
            num_slots: Slots including the reference.
            num_envs: Environments of the simulation, at least ``slice_size * num_slots``.
            step_dt: Duration of one environment step [s].
            device: Device of the environment tensors.
        """
        self._size, self._slots, self._dt = slice_size, num_slots, step_dt
        self._count = slice_size * num_slots
        self._returns = torch.zeros(self._count, device=device)
        self._lengths = torch.zeros(self._count, device=device)
        self._alive = torch.ones(self._count, dtype=torch.bool, device=device)
        self._episodes: list[deque[tuple[float, float]]] = [deque(maxlen=_WINDOW) for _ in range(num_slots)]
        self._active = [False] * num_slots

    def set_active(self, slot: int, active: bool) -> None:
        """Mark a slot as playing a checkpoint; a slot that was just filled or emptied starts with no statistics."""
        self._active[slot] = active
        self._episodes[slot].clear()

    def reset(self) -> None:
        """Forget unfinished episodes, after the environments were reset."""
        self._returns.zero_()
        self._lengths.zero_()
        self._alive.fill_(True)

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Add one environment step.

        Args:
            rewards: Reward of every environment, shape ``[num_envs]``.
            dones: Whether every environment terminated or timed out, shape ``[num_envs]``.
        """
        rewards, dones = rewards[: self._count].reshape(-1).float(), dones[: self._count].reshape(-1).bool()
        self._returns += rewards
        self._lengths += 1
        finished = dones & self._alive
        self._record(finished)
        self._alive &= ~dones
        reference = torch.nonzero(dones[: self._size]).reshape(-1)
        if reference.numel():
            # the reference reset, so every slot's counterpart starts over; a ghost still running ends here
            counterparts = reference[None, :] + torch.arange(self._slots, device=reference.device)[:, None] * self._size
            counterparts = counterparts.reshape(-1)
            running = torch.zeros_like(self._alive)
            running[counterparts] = True
            self._record(running & self._alive)
            self._returns[counterparts] = 0.0
            self._lengths[counterparts] = 0.0
            self._alive[counterparts] = True

    def metrics(self, slot: int) -> PolicyGhostMetrics | None:
        """Return a slot's statistics over its latest finished episodes, or None before it finished one."""
        episodes = self._episodes[slot]
        if not episodes:
            return None
        returns, lengths = zip(*episodes)
        return PolicyGhostMetrics(len(episodes), sum(returns) / len(returns), sum(lengths) / len(lengths))

    def _record(self, mask: torch.Tensor) -> None:
        for slot in range(self._slots):
            if not self._active[slot]:
                continue
            chunk = mask[slot * self._size : (slot + 1) * self._size]
            if chunk.any():
                returns = self._returns[slot * self._size : (slot + 1) * self._size][chunk].tolist()
                lengths = self._lengths[slot * self._size : (slot + 1) * self._size][chunk].tolist()
                self._episodes[slot].extend((r, n * self._dt) for r, n in zip(returns, lengths))
