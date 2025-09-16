# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .modifier_base import ModifierBase
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from . import modifier_cfg

##
# Modifiers as functions
##


def scale(data: torch.Tensor, multiplier: float) -> torch.Tensor:
    """Scales input data by a multiplier.

    Args:
        data: The data to apply the scale to.
        multiplier: Value to scale input by.

    Returns:
        Scaled data. Shape is the same as data.
    """
    return data * multiplier


def clip(data: torch.Tensor, bounds: tuple[float | None, float | None]) -> torch.Tensor:
    """Clips the data to a minimum and maximum value.

    Args:
        data: The data to apply the clip to.
        bounds: A tuple containing the minimum and maximum values to clip data to.
            If the value is None, that bound is not applied.

    Returns:
        Clipped data. Shape is the same as data.
    """
    return data.clip(min=bounds[0], max=bounds[1])


def bias(data: torch.Tensor, value: float) -> torch.Tensor:
    """Adds a uniform bias to the data.

    Args:
        data: The data to add bias to.
        value: Value of bias to add to data.

    Returns:
        Biased data. Shape is the same as data.
    """
    return data + value


##
# Sample of class based modifiers
##


class DigitalFilter(ModifierBase):
    r"""Modifier used to apply digital filtering to the input data.

    `Digital filters <https://en.wikipedia.org/wiki/Digital_filter>`_ are used to process discrete-time
    signals to extract useful parts of the signal, such as smoothing, noise reduction, or frequency separation.

    The filter can be implemented as a linear difference equation in the time domain. This equation
    can be used to calculate the output at each time-step based on the current and previous inputs and outputs.

    .. math::
         y_{i} = X B - Y A = \sum_{j=0}^{N} b_j x_{i-j} - \sum_{j=1}^{M} a_j y_{i-j}

    where :math:`y_{i}` is the current output of the filter. The array :math:`Y` contains previous
    outputs from the filter :math:`\{y_{i-j}\}_{j=1}^M` for :math:`M` previous time-steps. The array
    :math:`X` contains current :math:`x_{i}` and previous inputs to the filter
    :math:`\{x_{i-j}\}_{j=1}^N` for :math:`N` previous time-steps respectively.
    The filter coefficients :math:`A` and :math:`B` are used to design the filter. They are column vectors of
    length :math:`M` and :math:`N + 1` respectively.

    Different types of filters can be implemented by choosing different values for :math:`A` and :math:`B`.
    We provide some examples below.

    Examples
    ^^^^^^^^

    **Unit Delay Filter**

    A filter that delays the input signal by a single time-step simply outputs the previous input value.

    .. math:: y_{i} = x_{i-1}

    This can be implemented as a digital filter with the coefficients :math:`A = [0.0]` and :math:`B = [0.0, 1.0]`.

    **Moving Average Filter**

    A moving average filter is used to smooth out noise in a signal. It is similar to a low-pass filter
    but has a finite impulse response (FIR) and is non-recursive.

    The filter calculates the average of the input signal over a window of time-steps. The linear difference
    equation for a moving average filter is:

    .. math:: y_{i} = \frac{1}{N} \sum_{j=0}^{N} x_{i-j}

    This can be implemented as a digital filter with the coefficients :math:`A = [0.0]` and
    :math:`B = [1/N, 1/N, \cdots, 1/N]`.

    **First-order recursive low-pass filter**

    A recursive low-pass filter is used to smooth out high-frequency noise in a signal. It is a first-order
    infinite impulse response (IIR) filter which means it has a recursive component (previous output) in the
    linear difference equation.

    A first-order low-pass IIR filter has the difference equation:

    .. math:: y_{i} = \alpha y_{i-1} + (1-\alpha)x_{i}

    where :math:`\alpha` is a smoothing parameter between 0 and 1. Typically, the value of :math:`\alpha` is
    chosen based on the desired cut-off frequency of the filter.

    This filter can be implemented as a digital filter with the coefficients :math:`A = [-\alpha]` and
    :math:`B = [1 - \alpha]`.
    """

    def __init__(self, cfg: modifier_cfg.DigitalFilterCfg, data_dim: tuple[int, ...], device: str):
        """Initializes digital filter.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.

        Raises:
            ValueError: If filter coefficients are None.
        """
        # check that filter coefficients are not None
        if cfg.A is None or cfg.B is None:
            raise ValueError("Digital filter coefficients A and B must not be None. Please provide valid coefficients.")

        # initialize parent class
        super().__init__(cfg, data_dim, device)

        # assign filter coefficients and make sure they are column vectors
        self.A = torch.tensor(self._cfg.A, device=self._device).unsqueeze(1)
        self.B = torch.tensor(self._cfg.B, device=self._device).unsqueeze(1)

        # create buffer for input and output history
        self.x_n = torch.zeros(self._data_dim + (self.B.shape[0],), device=self._device)
        self.y_n = torch.zeros(self._data_dim + (self.A.shape[0],), device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets digital filter history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if env_ids is None:
            env_ids = slice(None)
        # reset history buffers
        self.x_n[env_ids] = 0.0
        self.y_n[env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies digital filter modification with a rolling history window inputs and outputs.

        Args:
            data: The data to apply filter to.

        Returns:
            Filtered data. Shape is the same as data.
        """
        # move history window for input
        self.x_n = torch.roll(self.x_n, shifts=1, dims=-1)
        self.x_n[..., 0] = data

        # calculate current filter value: y[i] = Y*A - X*B
        y_i = torch.matmul(self.x_n, self.B) - torch.matmul(self.y_n, self.A)
        y_i.squeeze_(-1)

        # move history window for output and add current filter value to history
        self.y_n = torch.roll(self.y_n, shifts=1, dims=-1)
        self.y_n[..., 0] = y_i

        return y_i


class Integrator(ModifierBase):
    r"""Modifier that applies a numerical forward integration based on a middle Reimann sum.

    An integrator is used to calculate the integral of a signal over time. The integral of a signal
    is the area under the curve of the signal. The integral can be approximated using numerical methods
    such as the `Riemann sum <https://en.wikipedia.org/wiki/Riemann_sum>`_.

    The middle Riemann sum is a method to approximate the integral of a function by dividing the area
    under the curve into rectangles. The height of each rectangle is the value of the function at the
    midpoint of the interval. The area of each rectangle is the width of the interval multiplied by the
    height of the rectangle.

    This integral method is useful for signals that are sampled at regular intervals. The integral
    can be written as:

    .. math::
        \int_{t_0}^{t_n} f(t) dt & \approx \int_{t_0}^{t_{n-1}} f(t) dt + \frac{f(t_{n-1}) + f(t_n)}{2} \Delta t

    where :math:`f(t)` is the signal to integrate, :math:`t_i` is the time at the i-th sample, and
    :math:`\Delta t` is the time step between samples.
    """

    def __init__(self, cfg: modifier_cfg.IntegratorCfg, data_dim: tuple[int, ...], device: str):
        """Initializes the integrator configuration and state.

        Args:
            cfg: Integral parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
        """
        # initialize parent class
        super().__init__(cfg, data_dim, device)

        # assign buffer for integral and previous value
        self.integral = torch.zeros(self._data_dim, device=self._device)
        self.y_prev = torch.zeros(self._data_dim, device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets integrator state to zero.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if env_ids is None:
            env_ids = slice(None)
        # reset history buffers
        self.integral[env_ids] = 0.0
        self.y_prev[env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies integral modification to input data.

        Args:
            data: The data to integrate.

        Returns:
            Integral of input signal. Shape is the same as data.
        """
        # integrate using middle Riemann sum
        self.integral += (data + self.y_prev) / 2 * self._cfg.dt
        # update previous value
        self.y_prev[:] = data

        return self.integral

class DelayedObservation(ModifierBase):
    r"""A modifier used to return a stochastically delayed (stale) version of
    another observation term. This can also be used to model multi-rate
    observations for non-sensor terms, e.g., pure MDP terms or proprioceptive terms.

    This modifier takes an existing observation term/function, pushes each new batched
    observation into a DelayBuffer, and returns an older sample according to a
    per-environment integer time-lag. Lags are drawn in [min_lag, max_lag],
    with an optional probability to *hold* the previous lag (to mimic repeated 
    frames). With 'update_period>0' (multi-rate), new lags are applied only
    on refresh ticks, which occur every update_period. Between refreshes the
    realised lag can increase at most by +1 (frame hold). This process is
    causal: the lag for each environment can only increase by 1 each step,
    ensuring that the returned observation is never older than the previous
    step's lagged observation.

    Shapes are preserved: the returned tensor has the exact shape of the wrapped
    term (``[num_envs, *obs_shape]``).

    Configuration (required nesting)
    --------------------------------
    Isaac Lab's manager **requires** class-based term params to be nested under
    the "_" key.
    
    Param keys:
        func (callable): The observation function to wrap. Must be callable
            with signature ``func(env, **func_params) -> torch.Tensor`` returning
            a batched tensor of shape ``[num_envs, ...]``.
        func_params (dict): Optional dict of keyword args to pass to `func`.
        min_lag (int): Minimum time-lag (in steps) to sample. Default 0.
        max_lag (int): Maximum time-lag (in steps) to sample. Default 3.
        per_env (bool): If True, sample a different lag for each environment.
            If False, use the same lag for all envs. Default True.
        hold_prob (float): Probability in [0, 1] of holding the previous lag
            instead of sampling a new one. Default 0.0 (always sample new).
        update_period (int): If > 0, apply new lags every `update_period`
            policy steps (models a lower sensor cadence). Between updates, the
            lag can increase by at most +1 each step (frame hold). If 0 (default),
            update every step.
        per_env_phase (bool): Only relevant if `update_period > 0`. If True,
            each environment has a different random phase offset for lag updates.
            If False, all envs update their lag simultaneously. Default True.
    """

    def __init__(self, cfg: modifier_cfg.DelayedObservationCfg, data_dim: tuple[int, ...], device: str):

        """Initialize the DelayedObservation modifier.

        Args:
            cfg: Configuration parameters.
        """
        # initialize parent class 
        super().__init__(cfg, data_dim, device)
        if cfg.min_lag < 0 or cfg.max_lag < cfg.min_lag:
            raise ValueError("StochasticDelay: require 0 <= min_lag <= max_lag.")
        if cfg.hold_prob < 0.0 or cfg.hold_prob > 1.0:
            raise ValueError("StochasticDelay: hold_prob must be in [0, 1].")
        if cfg.update_period < 0:
            raise ValueError("StochasticDelay: update_period must be non-negative.")
        if cfg.update_period > 0 and cfg.update_period > cfg.max_lag:
            raise ValueError("StochasticDelay: update_period must be <= max_lag.")

        # state
        self._buf = DelayBuffer(history_length=cfg.max_lag + 1, batch_size=data_dim[0], device=device)
        self._prev_realized_lags: torch.Tensor | None = None  # [N]
        self._phases: torch.Tensor | None = None              # [N] if multi-rate
        self._step: int = 0
        
        # prefill buffer with zeros so early delays are valid
        zeros = torch.zeros(data_dim, device=device)
        for _ in range(cfg.max_lag + 1):
            self._buf.compute(zeros)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the delay buffer and internal state. Since the DelayBuffer
        does not support partial resets, if env_ids is not None, only the
        previous lags for those envs are reset to zero, forcing the
        latest observation to be returned on the next call preventing
        observations from before the reset being returned.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if env_ids is None:
            self._buf.reset()
            self._prev_realized_lags = None
            self._phases = None
            self._step = 0
            # prefill again with zeros
            zeros = torch.zeros(self._data_dim, device=self._device)
            for _ in range(self._cfg.max_lag + 1):
                self._buf.compute(zeros)
        else:
            if self._prev_realized_lags is not None:
                self._prev_realized_lags[env_ids] = 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Add the current data to the delay buffer and return a stale sample
        according to the current lag for each environment.

        Args:
            data: The data to apply delay to.

        Returns:
            Delayed data. Shape is the same as data.
        """
        cfg = self._cfg
        self._step += 1

        # initialize phases for multi-rate on first use
        if cfg.update_period > 0 and self._phases is None:
            if cfg.per_env_phase:
                self._phases = torch.randint(0, cfg.update_period, (self._data_dim[0],), device=self._device)
            else:
                self._phases = torch.zeros(self._data_dim[0], dtype=torch.long, device=self._device)

        # sample desired lags in [min_lag, max_lag]
        if cfg.min_lag == cfg.max_lag:
            desired_lags = torch.full((self._data_dim[0],), cfg.max_lag, dtype=torch.long, device=self._device)
        else:
            desired_lags = torch.randint(cfg.min_lag, cfg.max_lag + 1, (self._data_dim[0],), device=self._device)

        if not cfg.per_env:
            desired_lags = torch.full_like(desired_lags, desired_lags[0])

        # optional: hold previous realized lag
        if cfg.hold_prob > 0.0 and self._prev_realized_lags is not None:
            hold_mask = torch.rand((self._data_dim[0],), device=self._device) < cfg.hold_prob
            desired_lags = torch.where(hold_mask, self._prev_realized_lags, desired_lags)

        # multi-rate update behavior
        if cfg.update_period > 0:
            refresh_mask = ((self._step - self._phases) % cfg.update_period) == 0
            if self._prev_realized_lags is None:
                realized_lags = desired_lags
            else:
                # between refreshes, lag can only increase by +1 (clamped)
                hold_realized_lags = (self._prev_realized_lags + 1).clamp(max=cfg.max_lag)
                realized_lags = torch.where(refresh_mask, desired_lags, hold_realized_lags)
        else:
            # every step: causal clamp (at most +1 step older)
            if self._prev_realized_lags is None:
                realized_lags = desired_lags
            else:
                realized_lags = torch.minimum(desired_lags, self._prev_realized_lags + 1)

        realized_lags = realized_lags.clamp(min=cfg.min_lag, max=cfg.max_lag)
        self._prev_realized_lags = realized_lags

        # return stale sample
        self._buf.set_time_lag(realized_lags)
        return self._buf.compute(data)
