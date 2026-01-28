# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse-specific simulation manager implementation.

This module provides the Omniverse/PhysX backend implementation of the simulation manager.
It handles all Omniverse-specific physics simulation lifecycle, callbacks, and physics views.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import carb
import omni.kit
import omni.physics.tensors
import omni.physx
import omni.timeline
import omni.usd
from pxr import PhysxSchema

from .simulation_manager import SimulationManagerBase
from .simulation_events import SimulationEvent

__all__ = ["PhysxSimulationManager"]


class PhysxSimulationManager(SimulationManagerBase):
    """Omniverse/PhysX implementation of the simulation manager.

    This class implements the SimulationManagerBase interface using Omniverse's
    PhysX physics engine, timeline system, and carb event dispatcher.
    """

    # ------------------------------------------------------------------
    # Omniverse Interfaces
    # ------------------------------------------------------------------

    _timeline = omni.timeline.get_timeline_interface()
    _message_bus = carb.eventdispatcher.get_eventdispatcher()
    _physx_interface = omni.physx.get_physx_interface()
    _physx_sim_interface = omni.physx.get_physx_simulation_interface()
    _carb_settings = carb.settings.get_settings()

    # ------------------------------------------------------------------
    # Omniverse-specific State
    # ------------------------------------------------------------------

    # Simulation views (PhysX tensor API)
    _view: omni.physics.tensors.SimulationView | None = None
    _view_warp: omni.physics.tensors.SimulationView | None = None

    # Physics scenes registry
    _physics_scene_apis: OrderedDict[str, Any] = OrderedDict()

    # ------------------------------------------------------------------
    # Compatibility Stub for Isaac Sim
    # ------------------------------------------------------------------

    class _SimulationManagerInterfaceStub:
        """Minimal stub for Isaac Sim compatibility.

        Isaac Sim's internal code may reference _simulation_manager_interface.
        This stub provides no-op implementations for those calls.
        """

        @staticmethod
        def reset():
            pass

        @staticmethod
        def get_simulation_time() -> float:
            try:
                return omni.physx.get_physx_interface().get_simulation_time()
            except Exception:
                return 0.0

        @staticmethod
        def is_simulating() -> bool:
            try:
                return omni.physx.get_physx_interface().is_simulating()
            except Exception:
                return False

        # No-ops for unused methods
        get_num_physics_steps = staticmethod(lambda: 0)
        is_paused = staticmethod(lambda: False)
        get_callback_iter = staticmethod(lambda: 0)
        set_callback_iter = staticmethod(lambda v: None)
        register_deletion_callback = staticmethod(lambda cb: None)
        register_physics_scene_addition_callback = staticmethod(lambda cb: None)
        deregister_callback = staticmethod(lambda id: False)
        enable_usd_notice_handler = staticmethod(lambda f: None)
        enable_fabric_usd_notice_handler = staticmethod(lambda s, f: None)
        is_fabric_usd_notice_handler_enabled = staticmethod(lambda s: False)
        get_sample_count = staticmethod(lambda: 0)
        get_all_samples = staticmethod(lambda: [])
        get_buffer_capacity = staticmethod(lambda: 1000)
        get_current_time = staticmethod(lambda: carb.RationalTime(-1, 1))
        get_simulation_time_at_time = staticmethod(lambda t: 0.0)
        get_sample_range = staticmethod(lambda: None)
        log_statistics = staticmethod(lambda: None)

    _simulation_manager_interface = _SimulationManagerInterfaceStub()

    # ------------------------------------------------------------------
    # Lifecycle Implementation
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls) -> None:
        """Initialize the manager and set up timeline callbacks."""
        cls._setup_callbacks()
        cls._track_physics_scenes()

    @classmethod
    def clear(cls) -> None:
        """Clear all state and callbacks."""
        # Notify assets to clean up (PRIM_DELETION with "/" = clear all)
        cls.dispatch_prim_deletion("/")

        # Clear handles (timeline subscriptions auto-cleanup, message bus observers just need deletion)
        cls._handles.clear()
        cls._callbacks.clear()

        # Invalidate views before clearing
        if cls._view:
            cls._view.invalidate()
            cls._view = None
        if cls._view_warp:
            cls._view_warp.invalidate()
            cls._view_warp = None

        # Reset state
        cls._warmup_needed = True
        cls._view_created = False
        cls._assets_loaded = True
        cls._physics_scene_apis.clear()

    @classmethod
    def initialize_physics(cls) -> None:
        """Warm-start physics and create simulation views."""
        if not cls._warmup_needed:
            return

        cls._physx_interface.force_load_physics_from_usd()
        cls._physx_interface.start_simulation()
        cls._physx_interface.update_simulation(cls.get_physics_dt(), 0.0)
        cls._physx_sim_interface.fetch_results()

        cls._message_bus.dispatch_event(SimulationEvent.PHYSICS_WARMUP.value, payload={})
        cls._warmup_needed = False
        cls._create_views()

    # ------------------------------------------------------------------
    # Simulation Views
    # ------------------------------------------------------------------

    @classmethod
    def get_physics_sim_view(cls) -> omni.physics.tensors.SimulationView | None:
        """Get the physics simulation view."""
        return cls._view

    @classmethod
    def get_warp_sim_view(cls) -> omni.physics.tensors.SimulationView | None:
        """Get the warp simulation view (Omniverse-specific).

        Returns:
            The warp simulation view, or None if not created.
        """
        return cls._view_warp

    # ------------------------------------------------------------------
    # Physics Configuration
    # ------------------------------------------------------------------

    @classmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds."""
        if cls._physics_scene_apis:
            api = list(cls._physics_scene_apis.values())[0]
            hz = api.GetTimeStepsPerSecondAttr().Get()
            return 1.0 / hz if hz else 0.0
        return 1.0 / 60.0

    @classmethod
    def get_physics_sim_device(cls) -> str:
        """Get the physics simulation device."""
        suppress = cls._carb_settings.get_as_bool("/physics/suppressReadback")
        if suppress and cls._is_gpu_enabled():
            device_id = max(0, cls._carb_settings.get_as_int("/physics/cudaDevice"))
            return f"cuda:{device_id}"
        return "cpu"

    @classmethod
    def set_physics_sim_device(cls, device: str) -> None:
        """Set the physics simulation device."""
        if "cuda" in device:
            parts = device.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else max(0, cls._carb_settings.get_as_int("/physics/cudaDevice"))
            cls._carb_settings.set_int("/physics/cudaDevice", device_id)
            cls._carb_settings.set_bool("/physics/suppressReadback", True)
            cls._set_gpu_dynamics(True)
            cls._enable_fabric(True)
        elif device == "cpu":
            cls._carb_settings.set_bool("/physics/suppressReadback", False)
            cls._set_gpu_dynamics(False)
        else:
            raise ValueError(f"Device must be 'cuda[:N]' or 'cpu', got '{device}'")

    # ------------------------------------------------------------------
    # Event Dispatch
    # ------------------------------------------------------------------

    @classmethod
    def dispatch_prim_deletion(cls, prim_path: str) -> None:
        """Dispatch prim deletion event."""
        cls._message_bus.dispatch_event(SimulationEvent.PRIM_DELETION.value, payload={"prim_path": prim_path})

    # ------------------------------------------------------------------
    # Callback Registration
    # ------------------------------------------------------------------

    @classmethod
    def register_callback(
        cls,
        callback: Callable,
        event: SimulationEvent,
        order: int = 0,
        name: str | None = None,
    ) -> int:
        """Register a callback for a simulation event.

        Args:
            callback: Function to call when event fires.
            event: The event type to listen for.
            order: Priority (lower = earlier).
            name: Optional name for the callback.

        Returns:
            Callback ID for deregistration.
        """
        cid = cls._next_callback_id()
        cb = cls._wrap_bound_method(callback)

        # Message bus events (generic lifecycle events)
        if event in (
            SimulationEvent.PHYSICS_WARMUP,
            SimulationEvent.PHYSICS_READY,
            SimulationEvent.POST_RESET,
            SimulationEvent.SIMULATION_VIEW_CREATED,
            SimulationEvent.PRIM_DELETION,
        ):
            cls._callbacks[cid] = cls._message_bus.observe_event(
                event_name=event.value, order=order, on_event=cb
            )

        # Physics step events (PhysX-specific)
        elif event == SimulationEvent.POST_PHYSICS_STEP:
            cls._callbacks[cid] = cls._physx_interface.subscribe_physics_on_step_events(
                lambda dt, c=cb: c(dt) if cls._view_created else None,
                pre_step=False,
                order=order,
            )

        elif event == SimulationEvent.PRE_PHYSICS_STEP:
            cls._callbacks[cid] = cls._physx_interface.subscribe_physics_on_step_events(
                lambda dt, c=cb: c(dt) if cls._view_created else None,
                pre_step=True,
                order=order,
            )

        # Timeline events (Omniverse timeline)
        elif event == SimulationEvent.TIMELINE_STOP:
            cls._callbacks[cid] = cls._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), cb, order=order, name=name
            )

        else:
            raise ValueError(f"Unsupported event: {event}")

        return cid

    @classmethod
    def enable_stage_open_callback(cls, enable: bool) -> None:
        """Enable or disable stage open tracking."""
        if enable and "stage_open" not in cls._handles:
            cls._handles["stage_open"] = cls._message_bus.observe_event(
                event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.OPENED),
                on_event=cls._on_stage_open,
            )
        elif not enable and "stage_open" in cls._handles:
            del cls._handles["stage_open"]
            cls._assets_loaded = True

    # ------------------------------------------------------------------
    # Internal: Callbacks Setup
    # ------------------------------------------------------------------

    @classmethod
    def _setup_callbacks(cls) -> None:
        """Set up internal timeline callbacks."""
        # Guard against duplicate subscriptions
        if "play" in cls._handles:
            return

        stream = cls._timeline.get_timeline_event_stream()
        cls._handles["play"] = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), cls._on_play
        )
        cls._handles["stop"] = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), cls._on_stop
        )
        cls.enable_stage_open_callback(True)

    @classmethod
    def _on_play(cls, event: Any) -> None:
        """Handle timeline play."""
        if cls._carb_settings.get_as_bool("/app/player/playSimulations"):
            cls.initialize_physics()

    @classmethod
    def _on_stop(cls, event: Any) -> None:
        """Handle timeline stop."""
        cls._warmup_needed = True
        if cls._view:
            cls._view.invalidate()
            cls._view = None
        if cls._view_warp:
            cls._view_warp.invalidate()
            cls._view_warp = None
        cls._view_created = False

    @classmethod
    def _on_stage_open(cls, event: Any) -> None:
        """Handle stage open."""
        cls._physics_scene_apis.clear()
        cls._callbacks.clear()
        cls._track_physics_scenes()
        cls._assets_loaded = True

        def on_loading(e):
            cls._assets_loaded = False

        def on_loaded(e):
            cls._assets_loaded = True

        ctx = omni.usd.get_context()
        cls._handles["assets_loading"] = cls._message_bus.observe_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADING),
            on_event=on_loading,
        )
        cls._handles["assets_loaded"] = cls._message_bus.observe_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
            on_event=on_loaded,
        )

    # ------------------------------------------------------------------
    # Internal: Simulation Views
    # ------------------------------------------------------------------

    @classmethod
    def _create_views(cls) -> None:
        """Create physics simulation views."""
        if cls._view_created:
            return

        from isaaclab.sim.utils.stage import get_current_stage_id

        stage_id = get_current_stage_id()
        cls._view = omni.physics.tensors.create_simulation_view(cls._backend, stage_id=stage_id)
        cls._view.set_subspace_roots("/")
        cls._view_warp = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
        cls._view_warp.set_subspace_roots("/")

        cls._physx_interface.update_simulation(cls.get_physics_dt(), 0.0)
        cls._view_created = True
        cls._message_bus.dispatch_event(SimulationEvent.SIMULATION_VIEW_CREATED.value, payload={})
        cls._message_bus.dispatch_event(SimulationEvent.PHYSICS_READY.value, payload={})

    # ------------------------------------------------------------------
    # Internal: Physics Scene Management
    # ------------------------------------------------------------------

    @classmethod
    def _track_physics_scenes(cls) -> None:
        """Scan stage for physics scenes."""
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim in stage.Traverse():
                if prim.GetTypeName() == "PhysicsScene":
                    cls._physics_scene_apis[prim.GetPath().pathString] = PhysxSchema.PhysxSceneAPI.Apply(prim)

    @classmethod
    def _is_gpu_enabled(cls) -> bool:
        """Check if GPU dynamics is enabled."""
        if cls._physics_scene_apis:
            api = list(cls._physics_scene_apis.values())[0]
            bp = api.GetBroadphaseTypeAttr().Get()
            gpu_dyn = api.GetEnableGPUDynamicsAttr().Get()
            return bp == "GPU" and gpu_dyn
        return False

    @classmethod
    def _set_gpu_dynamics(cls, enable: bool) -> None:
        """Enable/disable GPU dynamics on all physics scenes."""
        bp_type = "GPU" if enable else "MBP"
        for api in cls._physics_scene_apis.values():
            if not api.GetPrim().IsValid():
                continue
            if api.GetBroadphaseTypeAttr().Get() is None:
                api.CreateBroadphaseTypeAttr(bp_type)
            else:
                api.GetBroadphaseTypeAttr().Set(bp_type)
            if api.GetEnableGPUDynamicsAttr().Get() is None:
                api.CreateEnableGPUDynamicsAttr(enable)
            else:
                api.GetEnableGPUDynamicsAttr().Set(enable)

    @classmethod
    def _enable_fabric(cls, enable: bool) -> None:
        """Enable/disable physics fabric."""
        mgr = omni.kit.app.get_app().get_extension_manager()
        was_enabled = mgr.is_extension_enabled("omni.physx.fabric")
        if enable and not was_enabled:
            mgr.set_extension_enabled_immediate("omni.physx.fabric", True)
        elif not enable and was_enabled:
            mgr.set_extension_enabled_immediate("omni.physx.fabric", False)
        cls._carb_settings.set_bool("/physics/updateToUsd", not enable)
        cls._carb_settings.set_bool("/physics/updateParticlesToUsd", not enable)
        cls._carb_settings.set_bool("/physics/updateVelocitiesToUsd", not enable)
        cls._carb_settings.set_bool("/physics/updateForceSensorsToUsd", not enable)
