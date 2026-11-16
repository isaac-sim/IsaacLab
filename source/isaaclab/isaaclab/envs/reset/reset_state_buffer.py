# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-owned snapshots for masked reset-state restoration."""

from collections.abc import Callable
from functools import cache
from typing import TypeAlias

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

    def __init__(self, states: "ResetStateBuffer.State", *, num_states: int | None = None) -> None:
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
    def states(self) -> "ResetStateBuffer.State":
        """Copied :class:`ProxyArray` tree, preserving each field's shape, dtype, device, and units.

        This allocating export is intended for initialization, inspection, or persistence.
        """
        return _map_state(self._states, lambda value: ProxyArray(wp.clone(value, requires_grad=False)))

    def get_state(
        self,
        state_ids: wp.array | ProxyArray | torch.Tensor,
        *,
        out: "ResetStateBuffer.State | None" = None,
        mask: wp.array | ProxyArray | torch.Tensor | None = None,
    ) -> "ResetStateBuffer.State":
        """Copy selected states into the output rows enabled by ``mask``.

        Args:
            state_ids: Valid int32 or int64 catalog indices, shape [N], on the buffer device.
                Only IDs of enabled output rows are read.
            out: Preallocated tree with the stored schema and N leading rows. None allocates
                a new :class:`ProxyArray` tree and requires ``mask`` to be None.
            mask: Boolean selection, shape [N]. None copies every output row.

        Returns:
            The supplied output tree, or a new tree when ``out`` is None.
        """
        state_ids = _as_array(state_ids)
        if state_ids.ndim != 1 or state_ids.dtype not in (wp.int32, wp.int64):
            raise ValueError("state_ids must be a one-dimensional int32 or int64 array.")
        if mask is not None:
            mask = _as_array(mask)
            if mask.shape != state_ids.shape or mask.dtype != wp.bool or mask.device != state_ids.device:
                raise ValueError("mask must be a boolean array matching state_ids shape and device.")
            if out is None:
                raise ValueError("Masked reads require a preallocated out tree.")
        if self.num_states == 0 and state_ids.shape[0]:
            raise ValueError("Cannot read states from an empty buffer.")

        def read(states, target):
            allocate = target is None
            if allocate:
                target = {}
            elif not isinstance(target, dict) or target.keys() != states.keys():
                raise ValueError("out must match the stored state tree.")
            for name, value in states.items():
                if isinstance(value, dict):
                    target[name] = read(value, None if allocate else target[name])
                    continue
                shape = (state_ids.shape[0], *value.shape[1:])
                if allocate:
                    result = wp.empty(shape, dtype=value.dtype, device=value.device)
                    target[name] = ProxyArray(result)
                else:
                    result = _as_array(target[name], dtype=value.dtype)
                if (result.shape, result.dtype, result.device) != (shape, value.dtype, value.device):
                    raise ValueError(f"out field {name!r} must match the stored trailing shape, dtype, and device.")
                if state_ids.device != value.device:
                    raise ValueError("state_ids and stored states must share a device.")
                _gather(value, state_ids, mask, result)
            return target

        return read(self._states, out)


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
            _gather_kernel(source.dtype, source.ndim, ids.dtype),
            target.shape,
            [source, ids, mask, target],
            device=source.device,
        )


@cache
def _gather_kernel(dtype, ndim: int, id_dtype) -> wp.Kernel:
    @wp.kernel(module="unique")
    def gather(
        source: wp.array(dtype=dtype, ndim=ndim),
        ids: wp.array(dtype=id_dtype),
        mask: wp.array(dtype=wp.bool),
        target: wp.array(dtype=dtype, ndim=ndim),
    ):
        i, j, k, component = int(0), int(0), int(0), int(0)
        if wp.static(ndim == 1):
            i = wp.tid()
        elif wp.static(ndim == 2):
            i, j = wp.tid()
        elif wp.static(ndim == 3):
            i, j, k = wp.tid()
        else:
            i, j, k, component = wp.tid()
        if mask and not mask[i]:
            return
        row = int(ids[i])
        if wp.static(ndim == 1):
            target[i] = source[row]
        elif wp.static(ndim == 2):
            target[i, j] = source[row, j]
        elif wp.static(ndim == 3):
            target[i, j, k] = source[row, j, k]
        else:
            target[i, j, k, component] = source[row, j, k, component]

    return gather
