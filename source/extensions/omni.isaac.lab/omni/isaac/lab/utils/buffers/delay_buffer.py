# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed because we concatenate int and torch.Tensor in the type hints
from __future__ import annotations

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
    length of the underlying buffer, the most recent entry is returned.

    .. note::
        By default, the delay buffer has no delay, meaning that the data is returned as is.
    """

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        # set the parameters
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
        """The batch size of the ring buffer."""
        return self._circular_buffer.batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._circular_buffer.device

    @property
    def history_length(self) -> int:
        """The history length of the delay buffer.

        If zero, only the latest data is stored. If one, the latest and the previous data are stored, and so on.
        """
        return self._history_length

    @property
    def min_time_lag(self) -> int:
        """Minimum amount of time steps that can be delayed.

        This value cannot be negative or larger than :attr:`max_time_lag`.
        """
        return self._min_time_lag

    @property
    def max_time_lag(self) -> int:
        """Maximum amount of time steps that can be delayed.

        This value cannot be greater than :attr:`history_length`.
        """
        return self._max_time_lag

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag across each batch index.

        The shape of the tensor is (batch_size, ). The value at each index represents the delay for that index.
        This value is used to retrieve the data from the buffer.
        """
        return self._time_lags

    """
    Operations.
    """

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | None = None):
        """Sets the time lag for the delay buffer across the provided batch indices.

        Args:
            time_lag: The desired delay for the buffer.

              * If an integer is provided, the same delay is set for the provided batch indices.
              * If a tensor is provided, the delay is set for each batch index separately. The shape of the tensor
                should be (len(batch_ids),).

            batch_ids: The batch indices for which the time lag is set. Default is None, which sets the time lag
                for all batch indices.

        Raises:
            TypeError: If the type of the :attr:`time_lag` is not int or integer tensor.
            ValueError: If the minimum time lag is negative or the maximum time lag is larger than the history length.
        """
        # resolve batch indices
        if batch_ids is None:
            batch_ids = slice(None)

        # parse requested time_lag
        if isinstance(time_lag, int):
            # set the time lags across provided batch indices
            self._time_lags[batch_ids] = time_lag
        elif isinstance(time_lag, torch.Tensor):
            # check valid dtype for time_lag: must be int or long
            if time_lag.dtype not in [torch.int, torch.long]:
                raise TypeError(f"Invalid dtype for time_lag: {time_lag.dtype}. Expected torch.int or torch.long.")
            # set the time lags
            self._time_lags[batch_ids] = time_lag.to(device=self.device)
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or integer tensor.")

        # compute the min and max time lag
        self._min_time_lag = int(torch.min(self._time_lags).item())
        self._max_time_lag = int(torch.max(self._time_lags).item())
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError(f"The minimum time lag cannot be negative. Received: {self._min_time_lag}")
        if self._max_time_lag > self._history_length:
            raise ValueError(
                f"The maximum time lag cannot be larger than the history length. Received: {self._max_time_lag}"
            )

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        self._circular_buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append the input data to the buffer and returns a stale version of the data based on time lag delay.

        If the requested delay is larger than the number of buffered data points since the last reset,
        the function returns the latest data. For instance, if the delay is set to 2 and only one data point
        is stored in the buffer, the function will return the latest data. If the delay is set to 2 and three
        data points are stored, the function will return the first data point.

        Args:
           data: The input data. Shape is (batch_size, ...).

        Returns:
            The delayed version of the data from the stored buffer. Shape is (batch_size, ...).
        """
        # add the new data to the last layer
        self._circular_buffer.append(data)
        # return output
        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone()
