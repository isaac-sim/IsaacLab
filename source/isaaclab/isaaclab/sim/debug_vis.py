"""Helpers for backend-agnostic debug visualization callbacks."""

from __future__ import annotations

import weakref

from .simulation_context import SimulationContext


def should_use_visualizer_step_debug_vis() -> bool:
    """Return whether standalone visualizers should drive debug-vis callbacks."""

    sim = SimulationContext.instance()
    if sim is None:
        return False

    return any(
        viz.supports_markers() and not viz.pumps_app_update() and getattr(viz.cfg, "enable_markers", True)
        for viz in sim.visualizers
    )


def register_visualizer_step_debug_vis(owner, callback_name: str = "_debug_vis_callback"):
    """Register a callback on the SimulationContext visualizer-step hook."""

    sim = SimulationContext.instance()
    if sim is None:
        return None

    owner_ref = weakref.ref(owner)
    callback_id = f"{callback_name}:{type(owner).__name__}:{id(owner)}"

    def _callback(event):
        owner_obj = owner_ref()
        if owner_obj is not None:
            getattr(owner_obj, callback_name)(event)

    return sim.add_visualizer_callback(callback_id, _callback)
