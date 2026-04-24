# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import torch


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data.

    This class implements a circular buffer for storing a history of batched tensor data. The buffer is
    initialized with a maximum length and a batch size. The data is stored in a circular fashion, and the
    data can be retrieved in a LIFO (Last-In-First-Out) fashion. The buffer is designed to be used in
    multi-environment settings, where each environment has its own data.

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
        # the pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        # the actual buffer for data storage
        # note: this is initialized on the first call to :meth:`append`
        self._buffer: torch.Tensor = None  # type: ignore
        # CPU-side flag that mirrors ``any(self._num_pushes == 0)`` without
        # requiring a GPU→CPU synchronization in the hot path. It flips True
        # whenever :meth:`reset` marks any batch index for first-push
        # replication, and clears when the next :meth:`append` performs that
        # replication. Reads in :meth:`__getitem__` use this to raise the same
        # "buffer empty" error as before but without a per-call sync.
        self._any_first_push_pending: bool = False

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

        The shape of the buffer is (batch_size, max_length, ...).

        Note:
            The oldest entry is at the beginning of dimension 1.
        """
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    """
    Operations.
    """

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the circular buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        # resolve all indices
        if batch_ids is None:
            batch_ids = slice(None)
        # reset the number of pushes for the specified batch indices
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            # set buffer at batch_id reset indices to 0.0 so that the buffer()
            # getter returns the cleared circular buffer after reset.
            self._buffer[:, batch_ids, :] = 0.0
        # mark that at least one batch index now has ``num_pushes == 0`` so
        # the next :meth:`append` performs the first-push history replication
        # and :meth:`__getitem__` rejects reads until that append happens.
        self._any_first_push_pending = True

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
        # at the first call, initialize the buffer size
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
            # the buffer was just created, so every batch index starts with
            # ``num_pushes == 0`` and must be replicated on this first append
            self._any_first_push_pending = True
        # move the head to the next slot
        self._pointer = (self._pointer + 1) % self.max_length
        # add the new data to the last layer
        self._buffer[self._pointer] = data
        # Check for batches with zero pushes and initialize all values in
        # batch to first append. The CPU flag ``_any_first_push_pending``
        # mirrors ``torch.any(num_pushes == 0)`` but is maintained by
        # :meth:`reset` and cleared here, so we avoid a GPU→CPU sync every
        # append in the common case where no batch just reset.
        if self._any_first_push_pending:
            is_first_push = self._num_pushes == 0
            # Broadcast-safe write that works for arbitrary trailing data
            # shape. Equivalent to ``self._buffer[:, is_first_push] =
            # data[is_first_push]`` but without materializing the dynamic
            # boolean index (which would reintroduce a sync on some torch
            # versions via shape inference).
            mask = is_first_push.view(1, -1, *([1] * (data.ndim - 1)))
            self._buffer = torch.where(mask, data.unsqueeze(0), self._buffer)
            self._any_first_push_pending = False
        # increment number of number of pushes for all batches
        self._num_pushes += 1

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
        # check if the buffer is empty — equivalent to
        # ``torch.any(self._num_pushes == 0)`` but sync-free: the CPU flag
        # flips True in :meth:`reset` (or on buffer construction) and back to
        # False when :meth:`append` has filled every reset index's history.
        if self._any_first_push_pending or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        # admissible lag
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        # the index in the circular buffer (pointer points to the last+1 index)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        # return output
        return self._buffer[index_in_buffer, self._ALL_INDICES]
