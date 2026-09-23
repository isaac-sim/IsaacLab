# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared callable delay for measurements, actions, and actuator commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
import warp as wp

from .buffers import DelayBuffer
from .composition import WrapperCfg
from .configclass import configclass
from .types import ArticulationActions


@wp.kernel
def _schedule(
    lags: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int64),
    phase: wp.array(dtype=wp.int32),
    last_sample: wp.array(dtype=wp.int64),
    refresh: wp.array(dtype=wp.bool),
    counter: wp.array(dtype=wp.int64),
    seed: int,
    min_lag: int,
    max_lag: int,
    period: int,
    hold_prob: float,
    per_env: bool,
    resample: bool,
):
    i = wp.tid()
    stream = 0
    if per_env:
        stream = i
    rng = wp.rand_init(seed, wp.int32(counter[0]) * (lags.shape[0] + 1) + stream)
    if resample:
        lags[i] = wp.randi(rng, min_lag, max_lag + 1)
    sample = wp.max(wp.int64(0), steps[i] - wp.int64(lags[i]))
    deliver = (steps[i] - wp.int64(phase[i])) % wp.int64(period) == wp.int64(0)
    deliver = deliver and sample >= last_sample[i] and wp.randf(rng) >= hold_prob
    deliver = deliver or steps[i] == wp.int64(0)
    refresh[i] = deliver
    if deliver:
        last_sample[i] = sample


@wp.kernel
def _reset_schedule(
    indices: wp.array(dtype=wp.int64),
    lags: wp.array(dtype=wp.int32),
    phase: wp.array(dtype=wp.int32),
    last_sample: wp.array(dtype=wp.int64),
    resets: wp.array(dtype=wp.int64),
    seed: int,
    min_lag: int,
    max_lag: int,
    period: int,
    per_env: bool,
    per_env_phase: bool,
):
    i = indices[wp.tid()]
    stream = 0
    if per_env:
        stream = wp.int32(i)
    rng = wp.rand_init(seed, wp.int32(resets[i]) * (lags.shape[0] + 1) + stream)
    lags[i] = wp.randi(rng, min_lag, max_lag + 1)
    phase[i] = 0
    if per_env_phase and period > 1:
        phase_rng = wp.rand_init(seed, wp.int32(resets[i]) * (lags.shape[0] + 1) + wp.int32(i))
        phase[i] = wp.randi(phase_rng, 0, period)
    last_sample[i] = wp.int64(-1)
    resets[i] += wp.int64(1)


class Delay:
    """Wrap one complete evaluation with input or output latency.

    The enclosed callable still runs once per evaluation. Input delay leaves the other
    arguments, including live feedback, current. A command bundle shares one history and
    lag/hold decision. Returned data is independent of retained history.
    """

    def __init__(
        self,
        cfg: DelayCfg,
        term: Callable,
        num_envs: int,
        device: str,
        *,
        input_supported: bool = True,
        output_supported: bool = True,
        action: bool = False,
    ):
        wp.init()
        cfg.validate_config()
        if cfg.on == "input" and not input_supported or cfg.on == "output" and not output_supported:
            raise ValueError(f"This term does not expose a delayable {cfg.on} signal.")
        self._action = action
        self.cfg = cfg
        self.term = term
        self.device = device
        self._buffer = DelayBuffer(cfg.max_lag, num_envs, device)
        self._buffer.set_time_lag(cfg.min_lag)
        self._last_sample = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._phase = torch.zeros(num_envs, dtype=torch.int, device=device)
        self._refresh = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._counter = torch.zeros(1, dtype=torch.long, device=device)
        self._resets = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._all_indices = torch.arange(num_envs, device=device)
        self._seed = cfg.seed if cfg.seed is not None else int(torch.randint(2**31 - 1, ()).item())
        self._output: torch.Tensor | None = None
        self._packed: torch.Tensor | None = None
        self._fields: tuple[str, ...] | None = None
        self._reset_history(None)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None):
        """Reset selected histories and propagate reset through nested callable mechanisms."""
        reset = getattr(self.term, "reset", None)
        if reset is not None:
            reset(env_ids)
        self._reset_history(env_ids)

    def _reset_history(self, env_ids):
        self._buffer.reset(env_ids)
        indices = self._all_indices if env_ids is None else self._all_indices[env_ids]
        if not indices.numel():
            return
        wp.launch(
            _reset_schedule,
            dim=indices.numel(),
            device=self.device,
            stream=self._stream(),
            inputs=[
                wp.from_torch(indices),
                wp.from_torch(self._buffer.time_lags),
                wp.from_torch(self._phase),
                wp.from_torch(self._last_sample),
                wp.from_torch(self._resets),
                self._seed,
                self.cfg.min_lag,
                self.cfg.max_lag,
                self.cfg.update_period,
                self.cfg.per_env,
                self.cfg.per_env_phase,
            ],
        )

    def _stream(self):
        device = wp.get_device(self.device)
        if device.is_cuda:
            return wp.get_stream(device) if device.is_capturing else wp.stream_from_torch(self.device)
        return None

    def __call__(self, *args, **kwargs):
        # Actions stage policy input with arguments and evaluate physics output without arguments.
        if self._action and (bool(args) != (self.cfg.on == "input")):
            return self.term(*args, **kwargs)
        if self.cfg.on == "input":
            if not args:
                raise TypeError("Input delay requires the data signal as the first positional argument.")
            return self.term(self._delay(args[0]), *args[1:], **kwargs)
        return self._delay(self.term(*args, **kwargs))

    def _delay(self, data: torch.Tensor | ArticulationActions):
        if isinstance(data, torch.Tensor):
            if self.cfg.fields is not None:
                raise ValueError("fields selects command fields and cannot be used for a tensor signal.")
            return self._deliver(data)
        if not isinstance(data, ArticulationActions):
            raise TypeError(f"Delay requires a batched tensor or ArticulationActions, received {type(data)}.")
        fields = self.cfg.fields or ("position", "velocity", "feedforward_effort")
        names = {"position": "joint_positions", "velocity": "joint_velocities", "feedforward_effort": "joint_efforts"}
        selected = tuple(names[name] for name in fields if getattr(data, names[name]) is not None)
        if not selected:
            raise ValueError("The delayed command contains none of the selected fields.")
        values = [getattr(data, name) for name in selected]
        if self._fields is None:
            if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values):
                raise ValueError("Delayed command fields must share shape and dtype.")
            self._fields = selected
            self._packed = torch.empty((*values[0].shape, len(values)), dtype=values[0].dtype, device=self.device)
        elif self._fields != selected or any(
            value.shape != self._packed.shape[:-1] or value.dtype != self._packed.dtype for value in values
        ):
            raise ValueError("Delayed command fields, shape, and dtype cannot change after initialization.")
        for i, value in enumerate(values):
            self._packed[..., i].copy_(value)
        delivered = self._deliver(self._packed)
        result = ArticulationActions(
            data.joint_positions, data.joint_velocities, data.joint_efforts, data.joint_indices
        )
        for i, name in enumerate(selected):
            setattr(result, name, delivered[..., i])
        return result

    def _deliver(self, data: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.min_lag == cfg.max_lag and cfg.update_period == 1 and cfg.hold_prob == 0.0:
            return self._buffer(data)
        if self._output is None:
            # Allocate persistent state before ring-index temporaries can reuse its address during capture.
            self._output = torch.empty_like(data)
        wp.launch(
            _schedule,
            dim=self._buffer.batch_size,
            device=self.device,
            stream=self._stream(),
            inputs=[
                wp.from_torch(self._buffer.time_lags),
                wp.from_torch(self._buffer.num_pushes),
                wp.from_torch(self._phase),
                wp.from_torch(self._last_sample),
                wp.from_torch(self._refresh),
                wp.from_torch(self._counter),
                self._seed,
                cfg.min_lag,
                cfg.max_lag,
                cfg.update_period,
                cfg.hold_prob,
                cfg.per_env,
                cfg.resample == "step",
            ],
        )
        self._counter.add_(1)
        candidate = self._buffer(data)
        torch.where(self._refresh.view(-1, *([1] * (data.ndim - 1))), candidate, self._output, out=candidate)
        self._output.copy_(candidate)
        return candidate


@configclass
class DelayCfg(WrapperCfg):
    """Configure delay in evaluations of the wrapped term's declared signal.

    The first evaluation after reset delivers a fresh sample. Missing history uses the oldest
    sample from the current episode. Holds can make delivered data older than ``max_lag``.
    Actuator command delays count physics steps; observation delays count observations.
    """

    class_type: type = Delay
    on: Literal["input", "output"] = "output"
    """Whether to delay the first input signal or the returned output. Feedback arguments stay current."""
    fields: tuple[Literal["position", "velocity", "feedforward_effort"], ...] | None = None
    """Command fields to delay together. None selects all present fields; tensor signals need no selector."""
    min_lag: int = 0
    """Minimum sample age, measured in evaluations of the selected signal."""
    max_lag: int = 0
    """Maximum sampled age; also determines history capacity."""
    update_period: int = 1
    """Evaluations between delivery opportunities. Calls in between retain the previous output."""
    hold_prob: float = 0.0
    """Probability of retaining the previous output at a delivery opportunity."""
    per_env: bool = True
    """Sample lag and holds independently per environment rather than across the whole batch."""
    per_env_phase: bool = True
    """Sample an independent delivery phase per environment on reset; otherwise use phase zero."""
    resample: Literal["step", "reset"] = "step"
    """When to sample a new lag. Legacy actuator delays sample on reset."""
    seed: int | None = None
    """Independent device random stream seed. None draws a seed at construction."""

    def validate_config(self):
        if self.on not in ("input", "output") or self.resample not in ("step", "reset"):
            raise ValueError("Delay requires on='input' or 'output' and resample='step' or 'reset'.")
        if any(type(value) is not int for value in (self.min_lag, self.max_lag, self.update_period)):
            raise TypeError("Delay lags and update_period must be integers.")
        if not 0 <= self.min_lag <= self.max_lag or self.update_period < 1:
            raise ValueError("Delay requires 0 <= min_lag <= max_lag and a positive update_period.")
        if not 0.0 <= self.hold_prob <= 1.0:
            raise ValueError("Delay hold_prob must be in [0, 1].")
        if self.fields is not None and (
            not self.fields
            or len(set(self.fields)) != len(self.fields)
            or any(field not in ("position", "velocity", "feedforward_effort") for field in self.fields)
        ):
            raise ValueError("Delay fields must be unique position, velocity, or feedforward_effort names.")
