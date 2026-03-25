# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import torch


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data.

    This class stores a history of batched tensor data with the oldest entry at
    index 0 and the most recent entry at index ``max_len - 1`` of the internal
    buffer. The public indexing API remains LIFO (last-in-first-out), while the
    ordered internal layout keeps ``buffer`` retrieval cheap and makes the
    implementation compatible with tracing-based export flows.

    The shape of the appended data is expected to be (batch_size, ...), where the first dimension is the
    batch dimension. Correspondingly, the shape of the ring buffer is (max_len, batch_size, ...).
    """

    def __init__(self, max_len: int, batch_size: int, device: str):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum allowed value is 1.
            batch_size: The batch dimension of the data.
            device: The device used for processing.

        Raises:
            ValueError: If the buffer size is less than one.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        # set the parameters
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        # max length tensor for comparisons
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        # number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # the actual buffer for data storage
        # note: this is initialized on the first call to :meth:`append`
        self._buffer: torch.Tensor = None  # type: ignore

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
    def max_length(self) -> int:
        """The maximum length of the ring buffer."""
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        """The current length of the buffer. Shape is (batch_size,).

        Since the buffer is circular, the current length is the minimum of the number of pushes
        and the maximum length.
        """
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """Complete circular buffer with most recent entry at the end and oldest entry at the beginning.

        Returns:
            Complete circular buffer with most recent entry at the end and oldest entry at the beginning of
            dimension 1. The shape is [batch_size, max_length, data.shape[1:]].
        """
        return torch.transpose(self._buffer, dim0=0, dim1=1)

    """
    Operations.
    """

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the circular buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        batch_ids_resolved: Sequence[int] | slice
        if batch_ids is None:
            batch_ids_resolved = slice(None)
        else:
            batch_ids_resolved = batch_ids
        # reset the number of pushes for the specified batch indices
        self._num_pushes[batch_ids_resolved] = 0
        if self._buffer is not None:
            # set buffer at batch_id reset indices to 0.0 so that the buffer() getter returns
            # the cleared circular buffer after reset.
            self._buffer[:, batch_ids_resolved] = 0.0

    def append(self, data: torch.Tensor):
        """Append the data to the circular buffer.

        Args:
            data: The data to append to the circular buffer. The first dimension should be the batch dimension.
                Shape is (batch_size, ...).

        Raises:
            ValueError: If the input data has a different batch size than the buffer.
        """
        # check the batch size
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")

        # move the data to the device
        data = data.to(self._device)
        is_first_push = self._num_pushes == 0
        if self._buffer is None:
            self._buffer = data.unsqueeze(0).expand(self.max_length, *data.shape).clone()
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        # increment number of number of pushes for all batches
        self._append(data)
        self._num_pushes += 1

    def _append(self, data: torch.Tensor):
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=0)
        self._buffer[-1] = data

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """Retrieve the data from the circular buffer in last-in-first-out (LIFO) fashion.

        If the requested index is larger than the number of pushes since the last call to :meth:`reset`,
        the oldest stored data is returned.

        Args:
            key: The index to retrieve from the circular buffer. The index should be less than the number of pushes
                since the last call to :meth:`reset`. Shape is (batch_size,).

        Returns:
            The data from the circular buffer. Shape is (batch_size, ...).

        Raises:
            ValueError: If the input key has a different batch size than the buffer.
            RuntimeError: If the buffer is empty.
        """
        # check the batch size
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        if self._buffer is None:
            raise RuntimeError("The buffer is empty. Please append data before retrieving.")

        # admissible lag — clamp to [0, ..] so batches with _num_pushes == 0
        # return the zeroed-out slot instead of indexing out of bounds.
        valid_keys = torch.clamp(torch.minimum(key, self._num_pushes - 1), min=0)
        # The buffer is stored oldest->newest along dimension 0, so the most
        # recent item lives at the last index.
        index_in_buffer = (self.max_length - 1 - valid_keys).to(dtype=torch.long)
        # return output
        return self._buffer[index_in_buffer, self._ALL_INDICES]
