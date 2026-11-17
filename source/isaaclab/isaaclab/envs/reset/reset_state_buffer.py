# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-owned snapshots for masked reset-state restoration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray


class ResetStateBuffer:
    """Own a fixed set of task states independently of their generation and sampling.

    States are nested dictionaries of arrays sharing a leading row dimension and device.
    Warp composite types, such as transforms, retain their element type. Physical units
    follow the supplied fields; store positions relative to environment origins when
    sharing snapshots across environments.

    Construction and allocating accessors copy their inputs. For graph capture, allocate
    an output tree once, warm up :meth:`get_state`, then pass that tree and a full-size
    environment mask on every call. This path writes only selected environments and performs
    no array allocation or host synchronization. The task commits the arrays through its native
    asset writers.

    Args:
        states: Nested Warp, :class:`ProxyArray`, or Torch arrays, one state per leading row.
            An empty dictionary or zero-row arrays represent an empty buffer.
        num_states: Optional buffer size. A single supplied default state is repeated to
            fill this many rows. Otherwise it must match the supplied row count.

    Raises:
        ValueError: Arrays lack matching row counts and devices, or the requested size cannot
            be populated from the supplied states.
    """

    State: TypeAlias = dict[str, "wp.array | ProxyArray | torch.Tensor | ResetStateBuffer.State"]

    def __init__(self, states: ResetStateBuffer.State, *, num_states: int | None = None) -> None:
        if num_states is not None and (type(num_states) is not int or num_states < 0):
            raise ValueError("num_states must be a non-negative integer.")
        rows: int | None = None
        device: wp.context.Device | None = None
        repeated_ids = None

        def copy_array(value) -> wp.array:
            nonlocal rows, device, repeated_ids
            value = _as_array(value)
            if rows is None:
                rows, device = value.shape[0], value.device
            elif value.shape[0] != rows or value.device != device:
                raise ValueError("State arrays must share a leading row dimension and device.")
            if num_states is None or num_states == rows:
                return wp.clone(value, requires_grad=False)
            if rows != 1:
                raise ValueError("Only a single default state can be repeated to a different num_states.")
            if repeated_ids is None:
                repeated_ids = wp.zeros(num_states, dtype=wp.int32, device=device)
            result = wp.empty((num_states, *value.shape[1:]), dtype=value.dtype, device=device)
            _gather(value, repeated_ids, None, result)
            return result

        self._states = _map_state(states, copy_array)
        if rows is None and num_states not in (None, 0):
            raise ValueError("A non-empty schema is required to populate num_states.")
        self._num_states = (rows or 0) if num_states is None else num_states

    @property
    def num_states(self) -> int:
        """Number of stored states; zero for an empty buffer."""
        return self._num_states

    @property
    def states(self) -> ResetStateBuffer.State:
        """Copied :class:`ProxyArray` tree, preserving each field's shape, dtype, device, and units.

        This allocating export is intended for initialization, inspection, or persistence.
        """
        return _map_state(self._states, lambda value: ProxyArray(wp.clone(value, requires_grad=False)))

    def get_state(
        self,
        state_indices: wp.array | ProxyArray | torch.Tensor,
        *,
        out: ResetStateBuffer.State | None = None,
        mask: wp.array | ProxyArray | torch.Tensor | None = None,
    ) -> ResetStateBuffer.State:
        """Copy selected states into the output rows enabled by ``mask``.

        Invalid indices leave the corresponding output rows untouched, or zero-filled if allocating.

        Args:
            state_indices: Dense int32 or int64 row positions, shape [N], on the buffer device.
                Only indices of enabled output rows are read.
            out: Preallocated tree with the stored schema and N leading rows. None allocates
                a new :class:`ProxyArray` tree and requires ``mask`` to be None.
            mask: Boolean selection, shape [N]. None copies every output row.

        Returns:
            The supplied output tree, or a new tree when ``out`` is None.
        """
        state_indices = _as_array(state_indices)
        if state_indices.ndim != 1 or state_indices.dtype not in (wp.int32, wp.int64):
            raise ValueError("state_indices must be a one-dimensional int32 or int64 array.")
        if mask is not None:
            mask = _as_array(mask)
            if mask.shape != state_indices.shape or mask.dtype != wp.bool or mask.device != state_indices.device:
                raise ValueError("mask must be a boolean array matching state_indices shape and device.")
            if out is None:
                raise ValueError("Masked reads require a preallocated out tree.")
        num_rows = state_indices.shape[0]
        if self.num_states == 0 and num_rows:
            raise ValueError("Cannot read states from an empty buffer.")

        if out is None:
            out = _map_state(
                self._states,
                lambda value: ProxyArray(
                    wp.zeros((num_rows, *value.shape[1:]), dtype=value.dtype, device=value.device)
                ),
            )

        fields: list[tuple[wp.array, wp.array]] = []

        def collect_fields(states, target):
            if not isinstance(target, dict) or target.keys() != states.keys():
                raise ValueError("out must match the stored state tree.")
            for name, value in states.items():
                if isinstance(value, dict):
                    collect_fields(value, target[name])
                    continue
                shape = (num_rows, *value.shape[1:])
                result = _as_array(target[name], dtype=value.dtype)
                if (result.shape, result.dtype, result.device) != (shape, value.dtype, value.device):
                    raise ValueError(f"out field {name!r} must match the stored trailing shape, dtype, and device.")
                if state_indices.device != value.device:
                    raise ValueError("state_indices and stored states must share a device.")
                fields.append((value, result))

        collect_fields(self._states, out)
        for source, target in fields:
            _gather(source, state_indices, mask, target)
        return out


def _map_state(states: ResetStateBuffer.State, function: Callable) -> ResetStateBuffer.State:
    if not isinstance(states, dict):
        raise ValueError("states must be a nested dictionary of batched arrays.")
    return {
        key: _map_state(value, function) if isinstance(value, dict) else function(value)
        for key, value in states.items()
    }


def _as_array(value, dtype=None) -> wp.array:
    if isinstance(value, ProxyArray):
        value = value.warp
    elif isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise ValueError("State leaves must be arrays with a leading row dimension.")
        value = wp.from_torch(value, dtype=dtype)
    if not isinstance(value, wp.array) or value.ndim < 1:
        raise ValueError("State leaves must be arrays with a leading row dimension.")
    return value


def _gather(source, ids, mask, target) -> None:
    if target.size:
        wp.launch(
            _gather_kernel,
            target.shape,
            [source, ids, mask, target],
            device=source.device,
        )


@wp.kernel
def _gather_kernel(
    source: Any,
    ids: wp.array(dtype=Any),
    mask: wp.array(dtype=wp.bool),
    target: Any,
):
    i, j, k, component = int(0), int(0), int(0), int(0)
    if wp.static(len(source.shape) == 1):
        i = wp.tid()
    elif wp.static(len(source.shape) == 2):
        i, j = wp.tid()
    elif wp.static(len(source.shape) == 3):
        i, j, k = wp.tid()
    else:
        i, j, k, component = wp.tid()
    if mask and not mask[i]:
        return
    index = wp.int64(ids[i])
    if index < wp.int64(0) or index >= wp.int64(source.shape[0]):
        return
    row = int(index)
    if wp.static(len(source.shape) == 1):
        target[i] = source[row]
    elif wp.static(len(source.shape) == 2):
        target[i, j] = source[row, j]
    elif wp.static(len(source.shape) == 3):
        target[i, j, k] = source[row, j, k]
    else:
        target[i, j, k, component] = source[row, j, k, component]
