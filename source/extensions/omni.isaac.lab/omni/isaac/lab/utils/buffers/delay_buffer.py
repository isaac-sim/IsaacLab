# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from .circular_buffer import BatchedCircularBuffer


class DelayBuffer:
    """Provides the functionality to simulate delays for a data stream.

    The delay can be set constant, using :meth:`set_time_lag` with an integer input or different per environment, using
    the same method with an integer tensor input. If the requested delay is larger than the number of buffered data
    points since the last reset, the :meth:`compute` will return the oldest stored data, which is obviously less delayed
    than expected. Internally, this class uses a circular buffer for better computation efficiency.
    """

    def __init__(self, max_num_histories: int, num_envs: int, device: str):
        """Initialize the Delay Buffer.

        By default all the environments will have no delay.

        Args:
            max_num_histories: The maximum number of time steps that the data will be buffered. It is recommended
            to set this value equal to the maximum number of time lags that are expected. The minimum value is zero.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
        """
        self._max_num_histories = max(0, max_num_histories)
        # the buffer size: current data plus the history length
        self._circular_buffer = BatchedCircularBuffer(self._max_num_histories + 1, num_envs, device)
        # the minimum and maximum lags across all environments.
        self._min_time_lag = 0
        self._max_time_lag = 0
        # the lags for each environment.
        self._time_lags = torch.zeros(num_envs, dtype=torch.int, device=device)

    def set_time_lag(self, time_lag: int | torch.Tensor):
        """Sets the time lags for each environment.

        Args:
           time_lag: A single integer will result in a fixed delay across all environments, while a tensor of integers
           with the size (num_envs, ) will set a different delay for each environment. This value cannot be larger than
           :meth:`max_num_histories`.
        """
        # parse requested time_lag
        if isinstance(time_lag, int):
            self._min_time_lag = time_lag
            self._max_time_lag = time_lag
            self._time_lags = torch.ones(self.num_envs, dtype=torch.int, device=self.device) * time_lag
        elif isinstance(time_lag, torch.Tensor):
            if time_lag.size() != torch.Size([
                self.num_envs,
            ]):
                raise TypeError(
                    f"Invalid size for time_lag: {time_lag.size()}. Expected torch.Size([{self.num_envs}])."
                )
            self._min_time_lag = torch.min(time_lag).item()
            self._max_time_lag = torch.max(time_lag).item()
            self._time_lags = time_lag.to(dtype=torch.int, device=self.device)
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or Tensor.")
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError("Minimum of `time_lag` cannot be negative!")
        if self._max_time_lag > self._max_num_histories:
            raise ValueError(f"Maximum of `time_lag` cannot be larger than {self._max_num_histories}!")

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the delay buffer.

        Args:
            env_ids: List of environment IDs to reset.
        """
        self._circular_buffer.reset(env_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Adds the data to buffer and returns a stale version of the data based on :meth:`time_lags`.

        If the requested delay is larger than the number of buffered data points since the last reset, the :meth:`compute`
        will return the oldest stored data, which is obviously less delayed than expected.

        Args:
           data: The input data. Shape is ``(num_envs, num_feature)``.
        Returns:
            The delayed version of the input data. Shape is ``(num_envs, num_feature)``.
        """
        # add the new data to the last layer
        self._circular_buffer.append(data)
        # return output
        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """Number of articulations in the view."""
        return self._circular_buffer.batch_size

    @property
    def device(self) -> str:
        """Device used for processing."""
        return self._circular_buffer.device

    @property
    def max_num_histories(self) -> int:
        """Maximum number of time steps that the data can buffered."""
        return self._max_num_histories

    @property
    def min_time_lag(self) -> int:
        """Minimum number of time steps that the data can be delayed. This value cannot be negative."""
        return self._min_time_lag

    @property
    def max_time_lag(self) -> int:
        """Maximum number of time steps that the data can be delayed. This value cannot be greater than :meth:`max_num_histories`."""
        return self._max_time_lag

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag for each environment. These values are between :meth:`mim_time_lag` and :meth:`max_time_lag`."""
        return self._time_lags
