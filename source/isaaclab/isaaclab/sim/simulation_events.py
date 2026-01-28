# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation lifecycle events.

These events are backend-agnostic and define the simulation lifecycle that any
physics backend must support.
"""

from enum import Enum


class SimulationEvent(Enum):
    """Events dispatched during simulation lifecycle.

    Any physics backend implementation must dispatch these events at the appropriate
    times to ensure proper integration with Isaac Lab's simulation infrastructure.

    Event Lifecycle:
        1. PHYSICS_WARMUP - Physics engine is warming up (loading from USD, initializing)
        2. SIMULATION_VIEW_CREATED - Tensor simulation views are ready
        3. PHYSICS_READY - Physics is fully initialized and ready for stepping
        4. PRE_PHYSICS_STEP - Called before each physics step (for applying actions)
        5. POST_PHYSICS_STEP - Called after each physics step (for reading state)
        6. POST_RESET - Called after simulation reset
        7. TIMELINE_STOP - Simulation timeline stopped
        8. PRIM_DELETION - A prim is being deleted (for cleanup)
    """

    # Initialization events
    PHYSICS_WARMUP = "sim.physics_warmup"
    """Physics engine is warming up. Dispatched before simulation views are created."""

    SIMULATION_VIEW_CREATED = "sim.simulation_view_created"
    """Simulation views are created. Tensor APIs are now available."""

    PHYSICS_READY = "sim.physics_ready"
    """Physics is fully initialized. Safe to start stepping."""

    # Step events
    PRE_PHYSICS_STEP = "sim.pre_physics_step"
    """Called before each physics step. Use this to apply actions/commands."""

    POST_PHYSICS_STEP = "sim.post_physics_step"
    """Called after each physics step. Use this to read simulation state."""

    # Lifecycle events
    POST_RESET = "sim.post_reset"
    """Called after simulation reset. Use this to reinitialize state."""

    TIMELINE_STOP = "sim.timeline_stop"
    """Simulation timeline has stopped."""

    PRIM_DELETION = "sim.prim_deletion"
    """A prim is being deleted. Payload contains 'prim_path' key."""


# Backward compatibility alias
IsaacEvents = SimulationEvent
