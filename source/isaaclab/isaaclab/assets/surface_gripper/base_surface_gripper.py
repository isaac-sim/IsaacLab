# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common command and lifecycle interface for surface-gripper assets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
import warp as wp


class BaseSurfaceGripper(ABC):
    """Control replicated grippers through staged commands and measured seal state.

    Commands above 0.3 close; commands below -0.3 open. The middle band retains
    the vacuum state. Measured state is -1/open, 0/closing, or 1/sealed. Backend implementations
    own acquisition, forces, and native lifecycle resources.
    """

    is_graphable: ClassVar[bool] = False

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of grippers."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> wp.array:
        """Measured scalar state, shape ``(num_instances,)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def command(self) -> wp.array:
        """Staged scalar commands, shape ``(num_instances,)``."""
        raise NotImplementedError

    @abstractmethod
    def set_grippers_command_index(
        self,
        states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Stage commands without applying them to the simulation.

        Args:
            states: Float32 commands of shape ``(len(env_ids),)``; shape
                ``(num_instances,)`` when ``full_data`` is true or all grippers are selected.
            env_ids: Selected grippers, in compact command order. ``None`` selects all.
            full_data: Whether commands contain every gripper and are indexed by ``env_ids``.
        """
        raise NotImplementedError

    @abstractmethod
    def set_grippers_command_mask(
        self, states: torch.Tensor | wp.array, env_mask: torch.Tensor | wp.array | None = None
    ) -> None:
        """Stage full-sized commands for only the masked grippers.

        Args:
            states: Float32 commands of shape ``(num_instances,)``.
            env_mask: Boolean mask of shape ``(num_instances,)``. ``None`` selects all.
        """
        raise NotImplementedError

    @abstractmethod
    def set_grippers_command(self, states: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        """Stage compact commands for selected indices.

        Args:
            states: Commands in selected-index order, or all commands if ``indices`` is ``None``.
            indices: Selected grippers. ``None`` selects all.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Release selected grippers without changing other worlds or physical poses.

        Args:
            env_ids: Selected grippers. ``None`` selects all unless ``env_mask`` is supplied.
            env_mask: Boolean mask of shape ``(num_instances,)``; takes precedence over ``env_ids``.
        """
        raise NotImplementedError

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Apply staged commands."""
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float) -> None:
        """Refresh measured state after a simulation step."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release callbacks and native resources; repeated calls are safe."""
        raise NotImplementedError
