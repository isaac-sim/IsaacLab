# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed because we concatenate int and torch.Tensor in the type hints
from __future__ import annotations

from collections.abc import Sequence

import torch

from ..array import index_fill_


class DelayBuffer:
    """Ring storage for delayed batched tensors, independent of actions or observations.

    Each updating call writes one frame and retrieves a per-batch delayed frame. Storage is allocated
    on the first call and never shifted. The write index and per-batch history lengths stay on the device,
    including during CUDA graph replay. Callers select lags explicitly, or enable buffer-owned sampling
    with ``hold_prob``. The caller determines when to record a sample.

    When recording, a lag of zero returns the current input. Until enough samples exist after initialization or reset,
    the oldest available sample is returned. Reset only invalidates the selected batches' history;
    no previous-episode data can be read, and the remaining batches continue uninterrupted.
    """

    def __init__(
        self, history_length: int, batch_size: int, device: str, *, min_lag: int = 0, hold_prob: float | None = None
    ):
        """Initialize the delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
            min_lag: Minimum lag to sample, with :attr:`history_length` as the inclusive maximum.
                Defaults to zero. Used when ``hold_prob`` enables automatic sampling.
            hold_prob: Probability of retaining the current lag on each recorded sample. Defaults to None,
                preserving lags selected externally through :meth:`set_time_lag`. Setting a probability enables
                independent per-batch sampling at initialization and reset: 1.0 keeps that lag until reset,
                and 0.0 resamples on every recorded sample. Holding a lag keeps latency constant as frames advance.
        """
        self._history_length = max(0, history_length)
        if type(min_lag) is not int or not 0 <= min_lag <= self._history_length:
            raise ValueError("min_lag must be an integer in [0, history_length].")
        if hold_prob is not None and not 0.0 <= hold_prob <= 1.0:
            raise ValueError("hold_prob must be in [0, 1].")
        self._min_lag = min_lag
        self._hold_prob = hold_prob
        self._batch_size = batch_size
        self._device = device
        self._buffer: torch.Tensor | None = None
        self._write_index = torch.zeros(1, dtype=torch.long, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._ALL_INDICES = torch.arange(batch_size, device=device)
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)
        if hold_prob is not None:
            self.reset()

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
            self._time_lags[batch_ids] = time_lag.to(device=self.device, dtype=self._time_lags.dtype)
        else:
            index_fill_(self._time_lags, batch_ids, time_lag)

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Automatically sampled lags are redrawn for those batches regardless of ``hold_prob``.
        Externally selected lags are preserved.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        indices = slice(None) if batch_ids is None else batch_ids
        index_fill_(self._num_pushes, indices, 0)
        if self._hold_prob is not None:
            self._time_lags[indices] = torch.randint(
                self._min_lag,
                self.history_length + 1,
                self._time_lags[indices].shape,
                dtype=self._time_lags.dtype,
                device=self.device,
            )

    def compute(
        self,
        data: torch.Tensor,
        *,
        update_history: bool = True,
        batch_ids: Sequence[int] | slice | None = None,
    ) -> torch.Tensor:
        """Return delayed data, optionally recording the input as a new sample.

        If the requested delay exceeds the available history since reset, returns the oldest available
        sample. The result is independent of the internal storage and may be modified by the caller.

        Args:
           data: The input data. Shape is (batch_size, ...), or (len(batch_ids), ...) for selected batches.
           update_history: Whether to record the input as a new sample. Defaults to True. If False,
               return the delayed sample relative to the latest recorded frame without modifying the buffer.
               Batches with no recorded sample since initialization or reset return the input instead.
           batch_ids: Batches to read or update, in input row order. None selects all batches.

        Returns:
            The delayed data, with the same shape as the input.
        """
        selected_batch_size = self.batch_size
        if batch_ids is not None:
            selected_batch_size = (
                len(range(self.batch_size)[batch_ids]) if isinstance(batch_ids, slice) else len(batch_ids)
            )
        if data.shape[0] != selected_batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{selected_batch_size}'")
        if selected_batch_size == 0:
            return data.to(device=self.device).clone()
        if self._buffer is None:
            if not update_history:
                return data.to(device=self.device).clone()
            self._buffer = torch.empty(
                (self.history_length + 1, self.batch_size, *data.shape[1:]), dtype=data.dtype, device=self.device
            )
            if batch_ids is not None:
                self._buffer.zero_()
        elif data.shape[1:] != self._buffer.shape[2:]:
            raise ValueError(f"Expected data dimensions {self._buffer.shape[2:]}, received {data.shape[1:]}.")

        data = data.to(device=self.device, dtype=self._buffer.dtype)
        selected_batches = slice(None) if batch_ids is None else batch_ids
        # Keep the shared write position until independently updated batches need separate positions.
        if batch_ids is not None and update_history and self._write_index.numel() == 1:
            self._write_index = self._write_index.expand(self.batch_size).clone()
        write_index = self._write_index if self._write_index.numel() == 1 else self._write_index[selected_batches]
        num_pushes = self._num_pushes[selected_batches]
        time_lags = self._time_lags[selected_batches]
        environment_indices = self._ALL_INDICES[selected_batches]
        if not update_history:
            lag = torch.minimum(time_lags, (num_pushes - 1).clamp_min(0))
            read_index = (write_index - 1 - lag) % (self.history_length + 1)
            has_history = (num_pushes > 0).view(selected_batch_size, *([1] * (data.ndim - 1)))
            return torch.where(has_history, self._buffer[read_index, environment_indices], data)

        if self._hold_prob is not None and self._hold_prob < 1.0:
            lags = torch.randint(
                self._min_lag,
                self.history_length + 1,
                (selected_batch_size,),
                dtype=self._time_lags.dtype,
                device=self.device,
            )
            if self._hold_prob > 0.0:
                resample = torch.rand(selected_batch_size, device=self.device) >= self._hold_prob
                lags = torch.where(resample, lags, time_lags)
            self._time_lags[selected_batches] = lags
            time_lags = lags

        if self._write_index.numel() == 1 and batch_ids is None:
            self._buffer.index_copy_(0, self._write_index, data.unsqueeze(0))
        else:
            self._buffer[write_index, environment_indices] = data
        lag = torch.minimum(time_lags, num_pushes)
        read_index = (write_index - lag) % (self.history_length + 1)
        result = self._buffer[read_index, environment_indices]
        if batch_ids is None:
            self._num_pushes.add_(1)
            self._write_index.add_(1).remainder_(self.history_length + 1)
        else:
            self._num_pushes[selected_batches] = num_pushes + 1
            self._write_index[selected_batches] = (write_index + 1) % (self.history_length + 1)
        return result
