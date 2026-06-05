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

    When ``stack_dim`` is set, the internal layout is rearranged so that :attr:`stacked`
    returns the K frames merged into the chosen dim as a free contiguous view; :meth:`__getitem__`
    is disabled in this mode.
    """

    def __init__(self, max_len: int, batch_size: int, device: str, stack_dim: int | None = None):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum allowed value is 1.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
            stack_dim: If set, the buffer arranges its internal storage so :attr:`stacked` returns
                the K stored frames merged into ``data.shape[stack_dim]`` of the appended data
                as a free contiguous view. Any non-zero dim index in the appended data is valid
                (positive or negative); ``0`` (the batch dim) is invalid. Range validation against
                the actual data rank is deferred to the first :meth:`append`. For example,
                ``stack_dim=-1`` on a ``(B, H, W, C)`` input stacks K frames along the channel
                dim, yielding :attr:`stacked` shape ``(B, H, W, K*C)``. Defaults to ``None``
                (legacy layout).

        Raises:
            ValueError: If the buffer size is less than one, or ``stack_dim == 0``.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        if stack_dim is not None and stack_dim == 0:
            raise ValueError("stack_dim must not be 0 (cannot stack along the batch dimension).")

        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        # CPU mirror of max_len; avoids a GPU sync via ``.item()`` on every property access.
        self._max_len_int: int = max_len
        # max length tensor for comparisons
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        # number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # CPU gate; lets ``append`` skip a ``torch.any`` GPU sync on the steady-state path.
        self._need_reset: bool = True
        # Lazily allocated on the first ``append``.
        self._buffer: torch.Tensor = None  # type: ignore

        self._stack_dim_arg: int | None = stack_dim
        # Normalized position of K in internal storage; set on first append. None == legacy mode.
        self._stack_dim_internal: int | None = None

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
        return self._max_len_int

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
        if self._stack_dim_internal is None:
            return torch.transpose(self._buffer, dim0=0, dim1=1)
        return torch.movedim(self._buffer, source=self._stack_dim_internal, destination=1)

    @property
    def stacked(self) -> torch.Tensor:
        """Buffer contents with K frames merged along the configured ``stack_dim``.

        Frames appear in oldest -> newest order along the merged dim. The result is a view of
        the internal storage; callers must not mutate it.

        Returns:
            View of shape ``(batch_size, *frame_shape)`` with ``frame_shape[stack_dim]`` multiplied by ``max_length``.

        Raises:
            RuntimeError: If ``stack_dim`` was not set at construction.
        """
        if self._stack_dim_internal is None:
            if self._stack_dim_arg is None:
                raise RuntimeError("stacked is only available when CircularBuffer was created with stack_dim set.")
            raise RuntimeError("stacked is not yet available: call append() at least once to initialize the buffer.")
        k_pos = self._stack_dim_internal
        s = self._buffer.shape
        return self._buffer.reshape(*s[:k_pos], s[k_pos] * s[k_pos + 1], *s[k_pos + 2 :])

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
        self._num_pushes[batch_ids_resolved] = 0
        self._need_reset = True
        if self._buffer is not None:
            # set buffer at batch_id reset indices to 0.0 so that the buffer() getter returns
            # the cleared circular buffer after reset.
            if self._stack_dim_internal is None:
                self._buffer[:, batch_ids_resolved] = 0.0
            else:
                self._buffer[batch_ids_resolved] = 0.0

    def append(self, data: torch.Tensor):
        """Append the data to the circular buffer.

        Args:
            data: The data to append to the circular buffer. The first dimension should be the batch dimension.
                Shape is (batch_size, ...).

        Raises:
            ValueError: If the input data has a different batch size than the buffer.
            IndexError: On the first call, if the configured ``stack_dim`` is invalid for the
                appended data's rank.
        """
        # check the batch size
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")

        data = data.to(self._device)

        if self._buffer is None:
            self._allocate_buffer(data)

        # Shift slots so the newest write lands at the last K slot. Iterating front-to-back
        # keeps adjacent-slot copies non-overlapping. Cheap at the typical frame-stack K=2-4.
        if self._stack_dim_internal is None:
            for i in range(self._max_len_int - 1):
                self._buffer[i].copy_(self._buffer[i + 1])
            self._buffer[-1] = data
        else:
            k_pos = self._stack_dim_internal
            k = self._max_len_int
            for i in range(k - 1):
                self._buffer.narrow(k_pos, i, 1).copy_(self._buffer.narrow(k_pos, i + 1, 1))
            self._buffer.narrow(k_pos, k - 1, 1).copy_(data.unsqueeze(k_pos))

        if self._need_reset:
            is_first_push = self._num_pushes == 0
            if torch.any(is_first_push):
                if self._stack_dim_internal is None:
                    self._buffer[:, is_first_push] = data[is_first_push]
                else:
                    self._buffer[is_first_push] = data[is_first_push].unsqueeze(self._stack_dim_internal)
            self._need_reset = False

        self._num_pushes += 1

    def _allocate_buffer(self, data: torch.Tensor) -> None:
        """Allocate the internal buffer and finalize the storage layout on first append."""
        if self._stack_dim_arg is None:
            self._buffer = torch.empty((self._max_len_int, *data.shape), dtype=data.dtype, device=self._device)
            return

        ndim = data.ndim
        k_pos = self._stack_dim_arg
        if k_pos < 0:
            k_pos += ndim
        if k_pos < 1 or k_pos >= ndim:
            raise IndexError(
                f"stack_dim={self._stack_dim_arg} resolves to position {k_pos} for data with"
                f" ndim={ndim}; must be in [1, {ndim - 1}] or [-{ndim - 1}, -1]."
            )
        self._stack_dim_internal = k_pos
        self._buffer = torch.empty(
            (*data.shape[:k_pos], self._max_len_int, *data.shape[k_pos:]),
            dtype=data.dtype,
            device=self._device,
        )

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
            NotImplementedError: If the buffer was created with ``stack_dim`` set.
        """
        if self._stack_dim_internal is not None:
            raise NotImplementedError(
                "Indexing via __getitem__ is not supported in stacked-output mode. Use .stacked or .buffer instead."
            )
        # check the batch size
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        if self._buffer is None:
            raise RuntimeError("The buffer is empty. Please append data before retrieving.")

        # Clamp to [0, ..] so batches with _num_pushes == 0 return the zeroed slot.
        valid_keys = torch.clamp(torch.minimum(key, self._num_pushes - 1), min=0)
        # The buffer is stored oldest->newest along dimension 0, so the most
        # recent item lives at the last index.
        index_in_buffer = (self._max_len_int - 1 - valid_keys).to(dtype=torch.long)
        return self._buffer[index_in_buffer, self._ALL_INDICES]
