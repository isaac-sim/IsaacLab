# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Return-based RSI reset sampling.

The motion timeline is cut into the motion command's bins. ReturnSampler measures
V[b], the discounted return the policy earns from bin b on its own rollouts, and
turns it into the reset distribution: a uniform explore share, a softmax focus on
the bins with low return, and a keep share on the bins the focus has concentrated
on, so a hard motion that is learned keeps some practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReturnSamplerCfg:
    gamma: float = 0.99
    """Discount of the measured return; sets v_max = max_step_reward / (1 - gamma)."""
    horizon: int = 20
    """Steps per update. Each env's last `horizon` steps turn into returns that
  bootstrap from V where the window ends."""
    memory: int = 256
    """Most samples a bin's V remembers: V is the exact mean of a bin's first
  `memory` samples; after that, earlier samples count as `memory` samples against
  each update's new ones, so V follows the current policy."""
    steepness: float = 25.0
    """Softmax gain on V / v_max."""
    explore: float = 0.5
    """Share of resets spread uniformly over all bins."""
    keep: float = 0.1
    """Share of resets on bins in proportion to the summed square of the focus each
  received: concentrated focus earns keep, focus spread thin earns almost none."""


class ReturnSampler:
    def __init__(
        self,
        cfg: ReturnSamplerCfg,
        bin_count: int,
        frame_count: int,
        num_envs: int,
        max_step_reward: float,
        device: str = "cpu",
    ) -> None:
        assert 0.0 < cfg.gamma < 1.0, f"gamma must be in (0, 1), got {cfg.gamma}"
        assert cfg.explore + cfg.keep < 1.0, "explore + keep must leave room for focus"
        self.cfg = cfg
        self.bin_count = bin_count
        self.frame_count = frame_count
        self.v_max = max_step_reward / (1.0 - cfg.gamma)
        # A bin reads v_max until it is first measured.
        self.value = torch.full((bin_count,), self.v_max, device=device)
        self.count = torch.zeros(bin_count, device=device)
        # Summed over the whole run, so float64 keeps the small per-bin terms exact.
        self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
        shape = (num_envs, cfg.horizon)
        self._frames = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._step = 0

    def _bins(self, frames: torch.Tensor) -> torch.Tensor:
        """Frame -> bin, as the motion command bins its adaptive-sampling failures."""
        return frames * self.bin_count // self.frame_count

    def record(
        self,
        frames: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> torch.Tensor | None:
        """Buffer one env step: each env's frame before the step, its reward, and how
        its episode ended, if it did (a fall terminates; reaching the clip's last frame
        and a time-out truncate). Every `horizon` steps, update V and return the new
        reset distribution; otherwise None."""
        col = self._step
        self._frames[:, col] = frames
        self._rewards[:, col] = rewards
        self._terminated[:, col] = terminated
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._step += 1
        if self._step < self.cfg.horizon:
            return None
        self._step = 0
        self._update()
        return self._distribution()

    def _dones(self) -> torch.Tensor:
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self) -> torch.Tensor:
        """Discounted return of every buffered step, computed backwards like
        gae_returns with lambda 1. Where an episode ends, and at the buffer's last step,
        the return continues from the value of the state after the step: none after a
        fall; the mean of V after the clip's last frame, whose successor has no bin on
        the timeline; V at the next frame's bin after a time-out or the buffer's end."""
        gamma = self.cfg.gamma
        rewards = self._rewards
        next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
        next_value = torch.where(
            self._terminated,
            0.0,
            torch.where(self._clip_end, self.value.mean(), self.value[self._bins(next_frames)]),
        )
        dones = self._dones()
        returns = torch.empty_like(rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.cfg.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self) -> None:
        """Fold the buffer into V. Each env's visits to one bin within one episode give
        one sample, their mean return; a bin's earlier samples count at most `memory`
        times."""
        returns = self._returns()
        dones = self._dones()
        num_envs, horizon = dones.shape
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
        key = (envs * horizon + episode) * self.bin_count + self._bins(self._frames)
        keys, inverse, visits = torch.unique(key.reshape(-1), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=returns.device)
        samples.scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.bin_count
        new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        self.count = self.count.clamp(max=self.cfg.memory) + new
        self.value = torch.where(new > 0, self.value + (total - new * self.value) / self.count, self.value)

    def _distribution(self) -> torch.Tensor:
        cfg = self.cfg
        focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        explore = cfg.explore / self.bin_count
        return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep
