# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base interface class for SimulationContext subsystems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation_context import SimulationContext


class Interface(ABC):
    """Base class for simulation subsystem interfaces.

    Provides a common lifecycle: __init__() -> reset() -> step()/forward() (repeated) -> close()
    """

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize interface with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context

    @abstractmethod
    def reset(self, soft: bool = False) -> None:
        """Reset the subsystem.

        Args:
            soft: If True, skip full reinitialization.
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Update state without stepping simulation."""
        pass

    @abstractmethod
    def step(self, render: bool = True) -> None:
        """Step the subsystem by one timestep.

        Args:
            render: Whether to render after stepping. Defaults to True.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    def play(self) -> None:
        """Handle simulation start."""
        pass

    def pause(self) -> None:
        """Handle simulation pause."""
        pass

    def stop(self) -> None:
        """Handle simulation stop."""
        pass
