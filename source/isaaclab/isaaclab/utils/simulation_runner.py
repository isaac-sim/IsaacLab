# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for running simulation loops with visualizer-agnostic is_running checks.

This module provides a unified interface for running simulation loops that works with both
Omniverse SimulationApp and standalone Newton/Rerun visualizers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


def is_simulation_running(simulation_app: Any | None, sim_context: SimulationContext | None = None) -> bool:
    """Check if the simulation should continue running.

    This function abstracts the is_running check for both Omniverse and standalone modes:
    - In Omniverse mode: delegates to SimulationApp.is_running()
    - In standalone mode with visualizers: checks if any visualizer window is still open
    - In headless mode (no visualizers): always returns True (use Ctrl+C or break to exit)

    Args:
        simulation_app: The SimulationApp instance from AppLauncher.app.
            This will be None when running in standalone mode (Newton/Rerun visualizers).
        sim_context: The SimulationContext instance (e.g., env.unwrapped.sim).
            Required for checking visualizer status in standalone mode.

    Returns:
        True if the simulation loop should continue, False otherwise.

    Example:

        .. code-block:: python

            from isaaclab.app import AppLauncher
            from isaaclab.utils import is_simulation_running, close_simulation

            app_launcher = AppLauncher(args_cli)
            simulation_app = app_launcher.app

            env = gym.make(args_cli.task, cfg=env_cfg)

            while is_simulation_running(simulation_app, env.unwrapped.sim):
                # ... step logic ...
                pass

            env.close()
            close_simulation(simulation_app)

    """
    if simulation_app is not None:
        # Omniverse mode - use SimulationApp's is_running
        return simulation_app.is_running()

    # Standalone mode - check visualizers if available
    if sim_context is not None and hasattr(sim_context, "_visualizers"):
        visualizers = sim_context._visualizers
        if visualizers:
            # Return True if at least one visualizer is still running
            return any(v.is_running() for v in visualizers)

    # Pure headless mode or no sim_context - always return True
    # The user should use Ctrl+C or a break condition to exit the loop
    return True


def close_simulation(simulation_app: Any | None) -> None:
    """Close the simulation app if running in Omniverse mode.

    This function should be called at the end of the script to properly clean up resources.
    In standalone mode, this is a no-op (visualizers are closed via SimulationContext).

    Args:
        simulation_app: The SimulationApp instance from AppLauncher.app, or None if standalone.

    Example:

        .. code-block:: python

            # At end of script
            env.close()
            close_simulation(simulation_app)

    """
    if simulation_app is not None:
        simulation_app.close()
