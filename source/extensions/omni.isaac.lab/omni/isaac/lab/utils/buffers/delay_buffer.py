# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence

from .circular_buffer import CircularBuffer


class DelayBuffer:
    """Delay buffer that allows retrieving stored data with delays.

    This class uses a batched circular buffer to store input data. Different to a standard circular buffer,
    which uses the LIFO (last-in-first-out) principle to retrieve the data, the delay buffer class allows
    retrieving data based on the lag set by the user. For instance, if the delay set inside the buffer
    is 1, then the second last entry from the stream is retrieved. If it is 2, then the third last entry
    and so on.

    The class supports storing a batched tensor data. This means that the shape of the appended data
    is expected to be (batch_size, ...), where the first dimension is the batch dimension. Correspondingly,
    the delay can be set separately for each batch index. If the requested delay is larger than the current
    length of the underlying buffer, the most recent entry is returned. This is obviously less delayed than
    expected.
    """

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the delay buffer.

        By default all the environments will have no delay.

        Args:
            history_length: The maximum number of time steps that the data will be buffered. It is recommended
            to set this value equal to the maximum number of time lags that are expected. The minimum value is zero.
            batch_size: Number of articulations in the view.
            device: Device used for processing.
        """
        self._history_length = max(0, history_length)
        # the buffer size: current data plus the history length
        self._circular_buffer = CircularBuffer(self._history_length + 1, batch_size, device)
        # the minimum and maximum lags across all environments.
        self._min_time_lag = 0
        self._max_time_lag = 0
        # the lags for each environment.
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """Number of articulations in the view."""
        return self._circular_buffer.batch_size

    @property
    def device(self) -> str:
        """Device used for processing."""
        return self._circular_buffer.device

    @property
    def history_length(self) -> int:
        """Maximum number of time steps that the data can buffered."""
        return self._history_length

    @property
    def min_time_lag(self) -> int:
        """Minimum number of time steps that the data can be delayed. This value cannot be negative."""
        return self._min_time_lag

    @property
    def max_time_lag(self) -> int:
        """Maximum number of time steps that the data can be delayed. This value cannot be greater than :meth:`history_length`."""
        return self._max_time_lag

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag for each environment. These values are between :meth:`mim_time_lag` and :meth:`max_time_lag`."""
        return self._time_lags

    """
    Operations.
    """

    def set_time_lag(self, time_lag: int | torch.Tensor):
        """Sets the time lags for each environment.

        Args:
           time_lag: A single integer will result in a fixed delay across all environments, while a tensor of integers
           with the size (batch_size, ) will set a different delay for each environment. This value cannot be larger than
           :meth:`history_length`.
        """
        # parse requested time_lag
        if isinstance(time_lag, int):
            self._min_time_lag = time_lag
            self._max_time_lag = time_lag
            self._time_lags = torch.ones(self.batch_size, dtype=torch.int, device=self.device) * time_lag
        elif isinstance(time_lag, torch.Tensor):
            if time_lag.size() != torch.Size([self.batch_size]):
                raise TypeError(
                    f"Invalid size for time_lag: {time_lag.size()}. Expected torch.Size([{self.batch_size}])."
                )
            self._min_time_lag = int(torch.min(time_lag).item())
            self._max_time_lag = int(torch.max(time_lag).item())
            self._time_lags = time_lag.to(dtype=torch.int, device=self.device)
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or Tensor.")
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError("Minimum of `time_lag` cannot be negative!")
        if self._max_time_lag > self._history_length:
            raise ValueError(f"Maximum of `time_lag` cannot be larger than {self._history_length}!")

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
           data: The input data. Shape is ``(batch_size, num_feature)``.
        Returns:
            The delayed version of the input data. Shape is ``(batch_size, num_feature)``.
        """
        # add the new data to the last layer
        self._circular_buffer.append(data)
        # return output
        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone()
