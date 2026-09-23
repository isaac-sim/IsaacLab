# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed because we concatenate int and torch.Tensor in the type hints
from __future__ import annotations

from collections.abc import Sequence

import torch


class DelayBuffer:
    """Ring storage for delayed batched tensors, independent of actions or observations.

    Each :meth:`compute` writes one frame and retrieves a per-batch delayed frame. Storage is allocated
    on the first call and never shifted. The write index and per-batch history lengths stay on the device,
    including during CUDA graph replay. Lag sampling and update cadence belong to the caller.

    A lag of zero returns the current input. Until enough samples exist after initialization or reset,
    the oldest available sample is returned. Reset only invalidates the selected batches' history;
    no previous-episode data can be read, and the remaining batches continue uninterrupted.
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
        self._history_length = max(0, history_length)
        self._batch_size = batch_size
        self._device = device
        self._buffer: torch.Tensor | None = None
        self._write_index = torch.zeros(1, dtype=torch.long, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._ALL_INDICES = torch.arange(batch_size, device=device)
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        return self._batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._device

    @property
    def history_length(self) -> int:
        """The history length of the delay buffer.

        If zero, only the latest data is stored. If one, the latest and the previous data are stored, and so on.
        """
        return self._history_length

    @property
    def num_pushes(self) -> torch.Tensor:
        """Number of frames written since each batch's last reset. Shape is (batch_size,).

        Callers may read this device tensor to schedule updates; they must not modify it.
        """
        return self._num_pushes

    @property
    def min_time_lag(self) -> int:
        """Minimum amount of time steps that can be delayed.

        This value cannot be negative or larger than :attr:`max_time_lag`.
        """
        return int(self._time_lags.min().item())

    @property
    def max_time_lag(self) -> int:
        """Maximum amount of time steps that can be delayed.

        This value cannot be greater than :attr:`history_length`.
        """
        return int(self._time_lags.max().item())

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag across each batch index.

        The shape of the tensor is (batch_size, ). The value at each index represents the delay for that index.
        This value is used to retrieve the data from the buffer. Call :meth:`set_time_lag` to validate
        external inputs. Callers generating bounded lags on the device may update this tensor in place,
        keeping every value in ``[0, history_length]`` without a device-to-host validation round trip.
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

        # Validate the requested values before changing the live configuration.
        if isinstance(time_lag, int):
            min_time_lag = max_time_lag = time_lag
        elif isinstance(time_lag, torch.Tensor):
            # check valid dtype for time_lag: must be int or long
            if time_lag.dtype not in [torch.int, torch.long]:
                raise TypeError(f"Invalid dtype for time_lag: {time_lag.dtype}. Expected torch.int or torch.long.")
            min_time_lag = int(time_lag.min().item()) if time_lag.numel() else 0
            max_time_lag = int(time_lag.max().item()) if time_lag.numel() else 0
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or integer tensor.")

        if min_time_lag < 0:
            raise ValueError(f"The minimum time lag cannot be negative. Received: {min_time_lag}")
        if max_time_lag > self._history_length:
            raise ValueError(f"The maximum time lag cannot be larger than the history length. Received: {max_time_lag}")

        if isinstance(time_lag, torch.Tensor):
            time_lag = time_lag.to(device=self.device, dtype=self._time_lags.dtype)
        self._time_lags[batch_ids] = time_lag

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        self._num_pushes[slice(None) if batch_ids is None else batch_ids] = 0

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append the input data to the buffer and returns a stale version of the data based on time lag delay.

        If the requested delay exceeds the available history since reset, returns the oldest available
        sample. The result is independent of the internal storage and may be modified by the caller.

        Args:
           data: The input data. Shape is (batch_size, ...).

        Returns:
            The delayed version of the data from the stored buffer. Shape is (batch_size, ...).
        """
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")
        if self._buffer is None:
            self._buffer = torch.empty((self.history_length + 1, *data.shape), dtype=data.dtype, device=self.device)
        elif data.shape != self._buffer.shape[1:]:
            raise ValueError(f"Expected data shape {self._buffer.shape[1:]}, received {data.shape}.")

        data = data.to(device=self.device, dtype=self._buffer.dtype)
        self._buffer.index_copy_(0, self._write_index, data.unsqueeze(0))
        lag = torch.minimum(self._time_lags, self._num_pushes)
        read_index = (self._write_index - lag) % (self.history_length + 1)
        result = self._buffer[read_index, self._ALL_INDICES]
        self._num_pushes.add_(1)
        self._write_index.add_(1).remainder_(self.history_length + 1)
        return result
