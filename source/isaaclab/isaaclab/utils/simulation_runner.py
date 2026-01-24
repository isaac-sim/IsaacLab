# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for running simulation loops with visualizer-agnostic is_running checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


def is_simulation_running(simulation_app: Any | None, sim_context: SimulationContext | None = None) -> bool:
    """Check if the simulation should continue running.

    - Omniverse mode: delegates to SimulationApp.is_running()
    - Standalone mode with visualizers: checks if any visualizer window is still open
    - Headless mode (no visualizers): always returns True (use Ctrl+C or break to exit)
    """
    if simulation_app is not None:
        return simulation_app.is_running()

    if sim_context is not None and hasattr(sim_context, "_visualizers"):
        visualizers = sim_context._visualizers
        if visualizers:
            return any(v.is_running() for v in visualizers)

    return True


def close_simulation(simulation_app: Any | None) -> None:
    """Close the simulation app if running in Omniverse mode."""
    if simulation_app is not None:
        simulation_app.close()
