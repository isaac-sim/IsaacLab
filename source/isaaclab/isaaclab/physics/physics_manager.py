# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for physics managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class PhysicsManager(ABC):
    """Abstract base class for physics simulation managers.

    Physics managers handle the lifecycle of a physics simulation backend,
    including initialization, stepping, and cleanup.

    Lifecycle: __init__() -> reset() -> step() (repeated) -> close()
    """

    @classmethod
    @abstractmethod
    def initialize(cls, sim_context: "SimulationContext") -> None:
        """Initialize the physics manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        pass

    @classmethod
    @abstractmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        pass

    @classmethod
    @abstractmethod
    def forward(cls) -> None:
        """Update kinematics without stepping physics (for rendering)."""
        pass

    @classmethod
    @abstractmethod
    def step(cls) -> None:
        """Step physics simulation by one timestep (physics only, no rendering)."""
        pass

    @classmethod
    @abstractmethod
    def close(cls) -> None:
        """Clean up physics resources."""
        pass

    @classmethod
    @abstractmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds."""
        pass

    @classmethod
    @abstractmethod
    def get_device(cls) -> str:
        """Get the physics simulation device."""
        pass

    @classmethod
    @abstractmethod
    def get_physics_sim_view(cls):
        """Get the physics simulation view."""
        pass

    @classmethod
    @abstractmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled."""
        pass
