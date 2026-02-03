# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for physics backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class PhysicsBackend(ABC):
    """Base class for physics simulation backends.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize backend with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context

    @abstractmethod
    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Update kinematics without stepping physics."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Step physics simulation by one timestep (physics only, no rendering)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up physics resources."""
        pass
