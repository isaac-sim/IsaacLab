# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .newton_manager import NewtonManager
from .physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class NewtonBackend(PhysicsBackend):
    """Newton physics backend wrapping NewtonManager."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize and configure Newton backend.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)
        cfg = self._sim.cfg

        # Set simulation parameters
        NewtonManager.set_simulation_dt(cfg.dt)
        NewtonManager._gravity_vector = cfg.gravity

        # Extract and apply solver settings from config
        to_dict = getattr(cfg, "to_dict", None)
        params = to_dict() if callable(to_dict) else {}
        newton_cfg = dict(params.get("newton_cfg", {})) if isinstance(params, dict) else {}
        NewtonManager.set_solver_settings(newton_cfg)

        # USD fabric sync only needed for OV rendering
        NewtonManager._clone_physics_only = "omniverse" not in self._sim._visualizer_interface._visualizers_str

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            NewtonManager.start_simulation()
            NewtonManager.initialize_solver()

    def forward(self) -> None:
        """Update articulation kinematics without stepping physics."""
        NewtonManager.forward_kinematics()

    def step(self) -> None:
        """Step physics simulation."""
        if self._sim.is_playing():
            NewtonManager.step()

    def close(self) -> None:
        """Clean up Newton physics resources."""
        NewtonManager.clear()
        self._initialized = False
