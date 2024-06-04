# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Sequence


class BatchedCircularBuffer:
    """Circular buffer for storing a history of batched tensor data."""

    def __init__(self, max_len: int, batch_size: int, device: str):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum value is one.
            batch_size: The batch dimension of the data.
            device: Device used for processing.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        self._max_len = max_len
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)
        # number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # the pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        # the circular buffer for data storage
        self._buffer: torch.Tensor | None = None  # the data buffer

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the circular buffer.

        Args:
            batch_ids: Elements to reset in the batch dimension.
        """
        # resolve all indices
        if batch_ids is None:
            batch_ids = self._ALL_INDICES
        self._num_pushes[batch_ids] = 0

    def append(self, data: torch.Tensor):
        """Append the data to the circular buffer.

        Args:
            data: The data to be appended, where `len(data) == self.batch_size`.
        """
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has {data.shape[0]} environments while expecting {self.batch_size}")
        # at the fist call, initialize the buffer
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_len, *data.shape), dtype=data.dtype, device=self._device)
        # move the head to the next slot
        self._pointer = (self._pointer + 1) % self.max_len
        # add the new data to the last layer
        self._buffer[self._pointer] = data
        # increment number of number of pushes
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """Get the data from the circular buffer in LIFO fashion.

        Args:
            key: The index of the data to be retrieved. It can be a single integer or a tensor of integers.
        """
        if len(key) != self.batch_size:
            raise ValueError(f"The key has length {key.shape[0]} while expecting {self.batch_size}")
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise ValueError("Attempting to get data on an empty circular buffer.")
        # admissible lag
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        # the index in the circular buffer (pointer points to the last+1 index)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_len)
        # return output
        return self._buffer[index_in_buffer, self._ALL_INDICES, :]

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """The batch size in the ring buffer."""
        return self._batch_size

    @property
    def device(self) -> str:
        """Device used for processing."""
        return self._device

    @property
    def max_len(self) -> int:
        """The maximum length of the ring buffer."""
        return self._max_len
