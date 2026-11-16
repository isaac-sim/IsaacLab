# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed tensor snapshots for restoring selected reset states."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch


class ResetStateBuffer:
    """Own a fixed set of task states independently of their generation and sampling.

    States are nested dictionaries of tensors sharing a leading row dimension and device,
    compatible with :meth:`isaaclab.scene.InteractiveScene.get_state`. Physical units follow
    the supplied state fields. Capture scene states relative to environment origins when
    sharing the same buffer across environments.

    Construction detaches and copies the supplied tensors. Accessors return copies, so edits
    to the inputs or returned states cannot change the stored snapshots. The task restores
    selected states through its existing scene and controller APIs.

    Args:
        states: Non-empty tensor tree containing one state per leading row.

    Raises:
        ValueError: The tree has no tensor leaves, or leaves lack matching row counts and devices.
    """

    State: TypeAlias = dict[str, "torch.Tensor | ResetStateBuffer.State"]

    def __init__(self, states: ResetStateBuffer.State) -> None:
        rows: int | None = None
        device: torch.device | None = None

        def copy_tensor(value: torch.Tensor) -> torch.Tensor:
            nonlocal rows, device
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise ValueError("State leaves must be tensors with a leading row dimension.")
            if rows is None:
                rows, device = value.shape[0], value.device
            elif value.shape[0] != rows or value.device != device:
                raise ValueError("State tensors must share a leading row dimension and device.")
            return value.detach().clone()

        self._states = _map_state(states, copy_tensor)
        if rows is None:
            raise ValueError("states must be a non-empty tree of batched tensors.")
        self._num_states = rows

    @property
    def num_states(self) -> int:
        """Number of stored states, derived from the tensors' leading dimension."""
        return self._num_states

    @property
    def states(self) -> ResetStateBuffer.State:
        """Copy of the complete tensor tree, preserving each field's shape, dtype, device, and units.

        Use :meth:`get_state` to copy only selected rows during reset.
        """
        return _map_state(self._states, lambda value: value.clone())

    def get_state(self, state_ids: torch.Tensor) -> ResetStateBuffer.State:
        """Return copies of selected rows in the supplied order.

        Args:
            state_ids: Indices into this fixed tensor catalog, shape [N], on the buffer device.

        Returns:
            Tensor tree with a leading dimension of N and the captured fields' units.
        """
        return _map_state(self._states, lambda value: value[state_ids])


def _map_state(
    states: ResetStateBuffer.State, function: Callable[[torch.Tensor], torch.Tensor]
) -> ResetStateBuffer.State:
    if not isinstance(states, dict):
        raise ValueError("states must be a nested dictionary of batched tensors.")
    return {
        key: _map_state(value, function) if isinstance(value, dict) else function(value)
        for key, value in states.items()
    }
