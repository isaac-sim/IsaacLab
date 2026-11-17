# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common command and lifecycle interface for surface-gripper assets."""

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
        self, states: torch.Tensor | wp.array, env_ids: wp.array | None = None, full_data: bool = False
    ) -> None:
        """Stage compact commands, or full-sized commands when ``full_data`` is true."""
        raise NotImplementedError

    @abstractmethod
    def set_grippers_command_mask(self, states: torch.Tensor | wp.array, env_mask: torch.Tensor | None = None) -> None:
        """Stage full-sized commands for only the masked grippers."""
        raise NotImplementedError

    @abstractmethod
    def set_grippers_command(self, states: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        """Stage commands for selected indices; ``None`` selects every gripper."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None, env_mask: wp.array | None = None) -> None:
        """Release selected grippers without changing other worlds or physical poses."""
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
