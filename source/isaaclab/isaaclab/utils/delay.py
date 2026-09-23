# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared delay configuration and tensor delivery policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any

import torch

from .buffers import DelayBuffer
from .configclass import configclass


@configclass
class DelayCfg:
    """Wrap a callable with delay and sample-hold configuration.

    For an observation, use ``ObservationTermCfg(func=DelayCfg(term=func, params={...}, max_lag=3))``.
    The manager constructs the wrapped callable and owns its lifecycle. Configuration holds no runtime buffers.
    Lags and periods count evaluations of the wrapped term.
    """

    term: Any = MISSING
    """Wrapped callable, callable class, or its import path."""

    params: dict[str, Any] = dict()
    """Arguments to the wrapped callable. Put observation parameters here, not on the outer term."""

    min_lag: int = 0
    """Minimum sampled lag, inclusive."""

    max_lag: int = 0
    """Maximum sampled lag, inclusive. Determines the ring-buffer capacity."""

    update_period: int = 1
    """Number of calls between refresh opportunities. Must be positive."""

    hold_prob: float = 0.0
    """Probability of holding the last delivered frame at a refresh opportunity."""

    per_env: bool = True
    """Whether to sample independent lags for each environment."""

    per_env_phase: bool = True
    """Whether to randomize refresh phases per environment on reset."""

    def validate_config(self):
        if any(not isinstance(value, int) for value in (self.min_lag, self.max_lag, self.update_period)):
            raise TypeError("Delay lags and update_period must be integers.")
        if not 0 <= self.min_lag <= self.max_lag:
            raise ValueError("Delay requires 0 <= min_lag <= max_lag.")
        if self.update_period < 1:
            raise ValueError("Delay update_period must be positive.")
        if not 0.0 <= self.hold_prob <= 1.0:
            raise ValueError("Delay hold_prob must be in [0, 1].")


class _Delay:
    """Apply latency and sample holds to any batched tensor using :class:`DelayBuffer`.

    Each call stores the current input and samples a lag in ``[min_lag, max_lag]``. A refresh delivers
    that sample only if it is at least as recent as the last delivered sample. Between refreshes, or
    when a probabilistic hold occurs, the output stays unchanged. Held frames may therefore be older
    than ``max_lag``; that bound describes sampled transport latency, not time spent holding a frame.

    The first call after a full or partial reset delivers the new input immediately. During warmup,
    unavailable history resolves to the first input of the current episode. Returns independent storage
    so downstream clipping or scaling cannot change the held frame.

    """

    def __init__(self, cfg: DelayCfg, batch_size: int, device: str):
        cfg.validate_config()
        self._cfg = cfg
        self._device = device
        self._buffer = DelayBuffer(cfg.max_lag, batch_size, device)
        self._buffer.set_time_lag(cfg.min_lag)
        self._sample_step = torch.full_like(self._buffer.num_pushes, -1)
        self._phase = torch.zeros_like(self._buffer.num_pushes)
        self._output: torch.Tensor | None = None
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Invalidate only the selected environments' history and refresh schedule."""
        self._buffer.reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._sample_step[env_ids] = -1
        if self._cfg.per_env_phase and self._cfg.update_period > 1:
            self._phase[env_ids] = torch.randint(
                self._cfg.update_period, self._phase[env_ids].shape, device=self._device
            )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Store one input frame and return the latest delivered frame."""
        cfg = self._cfg
        if cfg.min_lag == cfg.max_lag and cfg.update_period == 1 and cfg.hold_prob == 0.0:
            return self._buffer.compute(data)

        # Sampling within validated bounds can update device lags without host synchronization.
        if cfg.min_lag != cfg.max_lag:
            if cfg.per_env:
                self._buffer.time_lags.random_(cfg.min_lag, cfg.max_lag + 1)
            else:
                self._buffer.time_lags.copy_(torch.randint(cfg.min_lag, cfg.max_lag + 1, (1,), device=self._device))
        candidate = self._buffer.compute(data)
        step = self._buffer.num_pushes - 1
        sample_step = (step - self._buffer.time_lags).clamp(min=0)
        refresh = ((step - self._phase) % cfg.update_period == 0) & (sample_step >= self._sample_step)
        if cfg.hold_prob > 0.0:
            refresh &= torch.rand(step.shape, device=self._device) >= cfg.hold_prob
        refresh |= step == 0
        if self._output is None:
            self._output = candidate.clone()
        else:
            torch.where(refresh.view(-1, *([1] * (data.ndim - 1))), candidate, self._output, out=candidate)
            self._output.copy_(candidate)
        torch.where(refresh, sample_step, self._sample_step, out=self._sample_step)
        return candidate
