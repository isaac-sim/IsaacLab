# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX Manager for Isaac Lab.

This module manages PhysX physics simulation lifecycle, configuration, callbacks, and physics views.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
import weakref
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import carb
import omni.kit
import omni.kit.app
import omni.physics.tensors
import omni.physx
import omni.timeline
import omni.usd
import torch
from pxr import PhysxSchema, Sdf

import isaaclab.sim as sim_utils
from .physics_manager import PhysicsManager

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext
    from .physx_manager_cfg import PhysxManagerCfg

__all__ = ["IsaacEvents", "PhysxManager"]

logger = logging.getLogger(__name__)


class IsaacEvents(Enum):
    """Events dispatched during simulation lifecycle."""

    PHYSICS_WARMUP = "isaac.physics_warmup"
    SIMULATION_VIEW_CREATED = "isaac.simulation_view_created"
    PHYSICS_READY = "isaac.physics_ready"
    POST_RESET = "isaac.post_reset"
    PRIM_DELETION = "isaac.prim_deletion"
    PRE_PHYSICS_STEP = "isaac.pre_physics_step"
    POST_PHYSICS_STEP = "isaac.post_physics_step"
    TIMELINE_STOP = "isaac.timeline_stop"


class AnimationRecorder:
    """Handles animation recording for the simulation.

    This class manages the recording of physics animations using the PhysX PVD
    (Physics Visual Debugger) interface. It handles the setup, update, and
    finalization of animation recordings.
    """

    def __init__(self, carb_settings: carb.settings.ISettings, app_iface: omni.kit.app.IApp):
        """Initialize the animation recorder.

        Args:
            carb_settings: The Carbonite settings interface.
            app_iface: The Omniverse Kit application interface.
        """
        self._carb_settings = carb_settings
        self._app_iface = app_iface
        self._enabled = False
        self._start_time: float = 0.0
        self._stop_time: float = 0.0
        self._started_timestamp: float | None = None
        self._output_dir: str = ""
        self._timestamp: str = ""
        self._physx_pvd_interface = None

        self._setup()

    @property
    def enabled(self) -> bool:
        """Whether animation recording is enabled."""
        return self._enabled

    def _setup(self) -> None:
        """Sets up animation recording settings and initializes the recording."""
        self._enabled = bool(self._carb_settings.get("/isaaclab/anim_recording/enabled"))
        if not self._enabled:
            return

        # Import omni.physx.pvd.bindings here since it is not available by default
        from omni.physxpvd.bindings import _physxPvd

        # Init anim recording settings
        self._start_time = self._carb_settings.get("/isaaclab/anim_recording/start_time")
        self._stop_time = self._carb_settings.get("/isaaclab/anim_recording/stop_time")
        self._started_timestamp = None

        # Make output path relative to repo path
        repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
        self._timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self._output_dir = (
            os.path.join(repo_path, "anim_recordings", self._timestamp).replace("\\", "/").rstrip("/") + "/"
        )
        os.makedirs(self._output_dir, exist_ok=True)

        # Acquire physx pvd interface and set output directory
        self._physx_pvd_interface = _physxPvd.acquire_physx_pvd_interface()

        # Set carb settings for the output path and enabling pvd recording
        self._carb_settings.set_string("/persistent/physics/omniPvdOvdRecordingDirectory", self._output_dir)
        self._carb_settings.set_bool("/physics/omniPvdOutputEnabled", True)

    def update(self) -> bool:
        """Tracks timestamps and triggers finish if total time has elapsed.

        Returns:
            True if animation recording has finished, False otherwise.
        """
        if not self._enabled:
            return False

        if self._started_timestamp is None:
            self._started_timestamp = time.time()

        total_time = time.time() - self._started_timestamp
        if total_time > self._stop_time:
            self._finish()
            return True
        return False

    def _finish(self) -> bool:
        """Finishes the animation recording and outputs the baked animation recording.

        Returns:
            True if the recording was successfully finished.
        """
        logger.warning(
            "[INFO][SimulationContext]: Finishing animation recording. Stage must be saved. Might take a few minutes."
        )

        # Detaching the stage will also close it and force the serialization of the OVD file
        physx = omni.physx.get_physx_simulation_interface()
        physx.detach_stage()

        # Save stage to disk
        stage_path = os.path.join(self._output_dir, "stage_simulation.usdc")
        sim_utils.save_stage(stage_path, save_and_reload_in_place=False)

        # Find the latest ovd file not named tmp.ovd
        ovd_files = [f for f in glob.glob(os.path.join(self._output_dir, "*.ovd")) if not f.endswith("tmp.ovd")]
        input_ovd_path = max(ovd_files, key=os.path.getctime)

        # Invoke pvd interface to create recording
        stage_filename = "baked_animation_recording.usda"
        result = self._physx_pvd_interface.ovd_to_usd_over_with_layer_creation(
            input_ovd_path,
            stage_path,
            self._output_dir,
            stage_filename,
            self._start_time,
            self._stop_time,
            True,  # True: ASCII layers / False : USDC layers
            False,  # True: verify over layer
        )

        # Workaround for manually setting the truncated start time in the baked animation recording
        self._update_usda_start_time(os.path.join(self._output_dir, stage_filename), self._start_time)

        # Disable recording
        self._carb_settings.set_bool("/physics/omniPvdOutputEnabled", False)

        return result

    @staticmethod
    def _update_usda_start_time(file_path: str, start_time: float) -> None:
        """Updates the start time of the USDA baked animation recording file.

        Args:
            file_path: Path to the USDA file.
            start_time: The new start time to set.
        """
        # Read the USDA file
        with open(file_path) as file:
            content = file.read()

        # Extract the timeCodesPerSecond value
        time_code_match = re.search(r"timeCodesPerSecond\s*=\s*(\d+)", content)
        if not time_code_match:
            raise ValueError("timeCodesPerSecond not found in the file.")
        time_codes_per_second = int(time_code_match.group(1))

        # Compute the new start time code
        new_start_time_code = int(start_time * time_codes_per_second)

        # Replace the startTimeCode in the file
        content = re.sub(r"startTimeCode\s*=\s*\d+", f"startTimeCode = {new_start_time_code}", content)

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(content)


class PhysxManager(PhysicsManager):
    """Manages PhysX physics simulation lifecycle, configuration, callbacks, and physics views.

    This is a class-level (singleton-like) manager for the PhysX simulation.
    It handles device settings (CPU/GPU), timestep/solver configuration,
    fabric interface for fast data access, physics stepping, and reset.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    # Core interfaces (names must match Isaac Sim's expected attributes)
    _timeline = omni.timeline.get_timeline_interface()
    _message_bus = carb.eventdispatcher.get_eventdispatcher()
    _physx_interface = omni.physx.get_physx_interface()
    _physx_sim_interface = omni.physx.get_physx_simulation_interface()
    _carb_settings = carb.settings.get_settings()

    # State
    _view: omni.physics.tensors.SimulationView | None = None
    _view_warp: omni.physics.tensors.SimulationView | None = None
    _backend: str = "torch"
    _warmup_needed: bool = True
    _view_created: bool = False
    _assets_loaded: bool = True

    # Physics scenes (name must match Isaac Sim's expected attribute)
    _physics_scene_apis: OrderedDict = OrderedDict()

    # Callbacks: id -> subscription
    _callbacks: dict = {}
    _callback_id: int = 0
    _handles: dict = {}  # Named internal handles

    # Simulation context reference (for stage, carb_settings, logger access)
    _sim: "SimulationContext | None" = None

    # Manager configuration (contains all physics settings)
    _cfg: "PhysxManagerCfg | None" = None

    # Device and fabric state
    _physics_device: str = "cpu"
    _fabric_iface = None
    _update_fabric = None
    _anim_recorder: AnimationRecorder | None = None

    # Compatibility stub for Isaac Sim code that calls _simulation_manager_interface
    class _PhysxManagerInterfaceStub:
        """Minimal stub for Isaac Sim compatibility."""

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

    _simulation_manager_interface = _PhysxManagerInterfaceStub()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, sim_context: "SimulationContext") -> None:
        """Initialize the manager with simulation context and set up physics.

        Args:
            sim_context: Parent simulation context (provides stage, carb_settings, logger).
        """
        cls._sim = sim_context
        # Store config reference for easy access
        cls._cfg = sim_context.cfg.physics_manager_cfg  # type: ignore[assignment]
        cls._setup_callbacks()
        cls._track_physics_scenes()

        # Configure physics settings
        cls._configure_simulation_dt()
        cls._apply_physics_settings()

        # a stage update here is needed for the case when physics_dt != rendering_dt
        cls._sim.set_setting("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()
        cls._sim.set_setting("/app/player/playSimulations", True)

        # load fabric interface
        cls._load_fabric_interface()

        # initialize animation recorder
        cls._anim_recorder = AnimationRecorder(cls._sim.carb_settings, omni.kit.app.get_app())

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset the physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            # initialize the physics simulation
            if cls._view is None:
                cls.initialize_physics()

        # app.update() may be changing the cuda device in reset, so we force it back to our desired device here
        if "cuda" in cls._physics_device:
            torch.cuda.set_device(cls._physics_device)

        # enable kinematic rendering with fabric
        if cls._view is not None:
            cls._view._backend.initialize_kinematic_bodies()

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics and fabric for rendering."""
        if cls._fabric_iface is not None and cls._update_fabric is not None:
            if cls._view is not None and cls._sim is not None and cls._sim.is_playing():
                # Update the articulations' link's poses before rendering
                cls._view.update_articulations_kinematic()
            cls._update_fabric(0.0, 0.0)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation (physics only, no rendering)."""
        if cls._cfg is None:
            return

        # update animation recorder if enabled
        if cls._anim_recorder is not None and cls._anim_recorder.enabled and cls._anim_recorder.update():
            logger.warning("Animation recording finished. Closing app.")
            omni.kit.app.get_app().shutdown()
            return

        # step physics only
        cls._physx_sim_interface.simulate(cls._cfg.dt, 0.0)
        cls._physx_sim_interface.fetch_results()

        # physics step may change cuda device, force it back
        if "cuda" in cls._physics_device:
            torch.cuda.set_device(cls._physics_device)

    @classmethod
    def close(cls) -> None:
        """Clean up physics resources."""
        # clear the simulation manager state (notifies assets to cleanup)
        cls.clear()
        # detach the stage from physx
        if cls._physx_sim_interface is not None:
            cls._physx_sim_interface.detach_stage()
        # clear references
        cls._sim = None
        cls._cfg = None
        cls._anim_recorder = None

    @classmethod
    def clear(cls) -> None:
        """Clear all state and callbacks."""
        # Notify assets to clean up (PRIM_DELETION with "/" = clear all)
        cls.dispatch_prim_deletion("/")
        # Properly unsubscribe handles before clearing
        # Timeline subscriptions are auto-cleaned by omni.timeline
        # Message bus observers just need to be deleted
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
        # Clear fabric interface
        cls._fabric_iface = None
        cls._update_fabric = None

    @classmethod
    def initialize_physics(cls) -> None:
        """Warm-start physics and create simulation views."""
        if not cls._warmup_needed:
            return
        cls._physx_interface.force_load_physics_from_usd()
        cls._physx_interface.start_simulation()
        cls._physx_interface.update_simulation(cls.get_physics_dt(), 0.0)
        cls._physx_sim_interface.fetch_results()
        cls._message_bus.dispatch_event(IsaacEvents.PHYSICS_WARMUP.value, payload={})
        cls._warmup_needed = False
        cls._create_views()

    @classmethod
    def get_physics_sim_view(cls) -> omni.physics.tensors.SimulationView | None:
        """Get the physics simulation view."""
        return cls._view

    @classmethod
    def get_backend(cls) -> str:
        """Get the tensor backend ("torch" or "warp")."""
        return cls._backend

    @classmethod
    def set_backend(cls, backend: str) -> None:
        """Set the tensor backend."""
        if backend not in ("torch", "warp"):
            raise ValueError(f"Backend must be 'torch' or 'warp', got '{backend}'")
        cls._backend = backend

    @classmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds."""
        # Prefer config if available
        if cls._cfg is not None:
            return cls._cfg.dt
        # Fallback to USD scene
        if cls._physics_scene_apis:
            api = list(cls._physics_scene_apis.values())[0]
            hz = api.GetTimeStepsPerSecondAttr().Get()
            return 1.0 / hz if hz else 0.0
        return 1.0 / 60.0

    @classmethod
    def get_device(cls) -> str:
        """Get the physics simulation device."""
        return cls._physics_device

    @classmethod
    def get_physics_sim_device(cls) -> str:
        """Get the physics simulation device (alias for get_device)."""
        return cls._physics_device

    @classmethod
    def is_fabric_enabled(cls) -> bool:
        """Returns whether the fabric interface is enabled."""
        return cls._fabric_iface is not None

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

    @classmethod
    def assets_loading(cls) -> bool:
        """Check if assets are currently loading."""
        return not cls._assets_loaded

    @classmethod
    def dispatch_prim_deletion(cls, prim_path: str) -> None:
        """Dispatch prim deletion event."""
        cls._message_bus.dispatch_event(IsaacEvents.PRIM_DELETION.value, payload={"prim_path": prim_path})

    # ------------------------------------------------------------------
    # Callback Registration
    # ------------------------------------------------------------------

    @classmethod
    def register_callback(cls, callback: Callable, event: IsaacEvents, order: int = 0, name: str | None = None) -> int:
        """Register a callback for a simulation event.

        Args:
            callback: Function to call when event fires.
            event: The event type to listen for.
            order: Priority (lower = earlier).
            name: Optional name for the callback.

        Returns:
            Callback ID for deregistration.
        """
        cid = cls._callback_id
        cls._callback_id += 1

        # Handle bound methods with weak references
        if hasattr(callback, "__self__"):
            obj_ref = weakref.proxy(callback.__self__)
            method_name = callback.__name__
            cb = lambda e, o=obj_ref, m=method_name: getattr(o, m)(e)
        else:
            cb = callback

        if event in (IsaacEvents.PHYSICS_WARMUP, IsaacEvents.PHYSICS_READY, IsaacEvents.POST_RESET,
                     IsaacEvents.SIMULATION_VIEW_CREATED, IsaacEvents.PRIM_DELETION):
            cls._callbacks[cid] = cls._message_bus.observe_event(event_name=event.value, order=order, on_event=cb)

        elif event == IsaacEvents.POST_PHYSICS_STEP:
            cls._callbacks[cid] = cls._physx_interface.subscribe_physics_on_step_events(
                lambda dt, c=cb: c(dt) if cls._view_created else None, pre_step=False, order=order
            )

        elif event == IsaacEvents.PRE_PHYSICS_STEP:
            cls._callbacks[cid] = cls._physx_interface.subscribe_physics_on_step_events(
                lambda dt, c=cb: c(dt) if cls._view_created else None, pre_step=True, order=order
            )

        elif event == IsaacEvents.TIMELINE_STOP:
            cls._callbacks[cid] = cls._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), cb, order=order, name=name
            )
        else:
            raise ValueError(f"Unsupported event: {event}")

        return cid

    @classmethod
    def deregister_callback(cls, callback_id: int) -> None:
        """Deregister a callback."""
        cls._callbacks.pop(callback_id, None)

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
    # Internal
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
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADING), on_event=on_loading
        )
        cls._handles["assets_loaded"] = cls._message_bus.observe_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED), on_event=on_loaded
        )

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
        cls._message_bus.dispatch_event(IsaacEvents.SIMULATION_VIEW_CREATED.value, payload={})
        cls._message_bus.dispatch_event(IsaacEvents.PHYSICS_READY.value, payload={})

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

    # ------------------------------------------------------------------
    # Physics Configuration (from PhysXBackend)
    # ------------------------------------------------------------------

    @classmethod
    def _configure_simulation_dt(cls) -> None:
        """Configures the physics simulation step size."""
        if cls._sim is None or cls._cfg is None:
            return

        carb_settings = cls._sim.carb_settings

        # Get physics scene API from the physics interface
        stage = cls._sim.stage
        physics_scene_prim = stage.GetPrimAtPath(cls._cfg.physics_prim_path)
        physx_scene_api = PhysxSchema.PhysxSceneAPI(physics_scene_prim)

        # if rendering is called the substeps term is used to determine
        # how many physics steps to perform per rendering step.
        # it is not used if step(render=False).
        render_interval = max(cls._sim.cfg.render_interval, 1)

        # set simulation step per second
        steps_per_second = int(1.0 / cls._cfg.dt)
        physx_scene_api.CreateTimeStepsPerSecondAttr(steps_per_second)
        # set minimum number of steps per frame
        min_steps = int(steps_per_second / render_interval)
        carb_settings.set_int("/persistent/simulation/minFrameRate", min_steps)

    @classmethod
    def _apply_physics_settings(cls) -> None:
        """Sets various carb physics settings."""
        if cls._sim is None or cls._cfg is None:
            return

        carb_settings = cls._sim.carb_settings

        # Get physics scene API from the physics interface
        stage = cls._sim.stage
        physics_scene_prim = stage.GetPrimAtPath(cls._cfg.physics_prim_path)
        physx_scene_api = PhysxSchema.PhysxSceneAPI(physics_scene_prim)

        # --------------------------
        # Carb Physics API settings
        # --------------------------

        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        carb_settings.set_bool("/physics/physxDispatcher", True)
        # disable contact processing in omni.physx
        # note: we disable it by default to avoid the overhead of contact processing when it isn't needed.
        #   The physics flag gets enabled when a contact sensor is created.
        # FIXME: From investigation, it seems this flag only affects CPU physics. For GPU physics, contacts
        #  are always processed. The issue is reported to the PhysX team by @mmittal.
        carb_settings.set_bool("/physics/disableContactProcessing", True)
        # disable custom geometry for cylinder and cone collision shapes to allow contact reporting for them
        # reason: cylinders and cones aren't natively supported by PhysX so we need to use custom geometry flags
        # reference: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html?highlight=capsule#geometry
        carb_settings.set_bool("/physics/collisionConeCustomGeometry", False)
        carb_settings.set_bool("/physics/collisionCylinderCustomGeometry", False)
        # hide the Simulation Settings window
        carb_settings.set_bool("/physics/autoPopupSimulationOutputWindow", False)

        # handle device settings
        if "cuda" in cls._cfg.device:
            parsed_device = cls._cfg.device.split(":")
            if len(parsed_device) == 1:
                # if users only specified "cuda", we check if carb settings provide a valid device id
                # otherwise, we default to device id 0
                device_id = carb_settings.get_as_int("/physics/cudaDevice")
                if device_id < 0:
                    carb_settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
            else:
                # if users specified "cuda:N", we use the provided device id
                device_id = int(parsed_device[1])
                carb_settings.set_int("/physics/cudaDevice", device_id)
            # suppress readback for GPU physics
            carb_settings.set_bool("/physics/suppressReadback", True)
            # save the device
            cls._physics_device = f"cuda:{device_id}"
        else:
            # enable USD read/write operations for CPU physics
            carb_settings.set_int("/physics/cudaDevice", -1)
            carb_settings.set_bool("/physics/suppressReadback", False)
            # save the device
            cls._physics_device = "cpu"

        # Configure simulation manager backend
        # Isaac Lab always uses torch tensors for consistency, even on CPU
        cls.set_backend("torch")
        cls.set_physics_sim_device(cls._physics_device)

        # --------------------------
        # USDPhysics API settings
        # --------------------------

        # create the default physics material
        # this material is used when no material is specified for a primitive
        if cls._cfg.physics_material is not None:
            material_path = f"{cls._cfg.physics_prim_path}/defaultMaterial"
            cls._cfg.physics_material.func(material_path, cls._cfg.physics_material)
            # bind the physics material to the scene
            sim_utils.bind_physics_material(cls._cfg.physics_prim_path, material_path)

        # --------------------------
        # PhysX API settings
        # --------------------------

        # set broadphase type
        broadphase_type = "GPU" if "cuda" in cls._cfg.device else "MBP"
        physx_scene_api.CreateBroadphaseTypeAttr(broadphase_type)
        # set gpu dynamics
        enable_gpu_dynamics = "cuda" in cls._cfg.device
        physx_scene_api.CreateEnableGPUDynamicsAttr(enable_gpu_dynamics)

        # GPU-dynamics does not support CCD, so we disable it if it is enabled.
        if enable_gpu_dynamics and cls._cfg.enable_ccd:
            cls._cfg.enable_ccd = False
            cls._sim.logger.warning(
                "CCD is disabled when GPU dynamics is enabled. Please disable CCD in the PhysxManagerCfg to avoid this"
                " warning."
            )
        physx_scene_api.CreateEnableCCDAttr(cls._cfg.enable_ccd)

        # set solver type
        solver_type = "PGS" if cls._cfg.solver_type == 0 else "TGS"
        physx_scene_api.CreateSolverTypeAttr(solver_type)

        # set solve articulation contact last
        attr = physx_scene_api.GetPrim().CreateAttribute(
            "physxScene:solveArticulationContactLast", Sdf.ValueTypeNames.Bool
        )
        attr.Set(cls._cfg.solve_articulation_contact_last)

        # iterate over all the settings and set them
        for key, value in cls._cfg.to_dict().items():  # type: ignore
            # Skip non-PhysX settings and already-handled settings
            if key in ["solver_type", "enable_ccd", "solve_articulation_contact_last",
                       "dt", "device", "render_interval", "gravity",
                       "physics_prim_path", "use_fabric", "physics_material"]:
                continue
            if key == "bounce_threshold_velocity":
                key = "bounce_threshold"
            sim_utils.safe_set_attribute_on_usd_schema(physx_scene_api, key, value, camel_case=True)

        # throw warnings for helpful guidance
        if cls._cfg.solver_type == 1 and not cls._cfg.enable_external_forces_every_iteration:
            logger.warning(
                "The `enable_external_forces_every_iteration` parameter in PhysxManagerCfg is set to False. If you are"
                " experiencing noisy velocities, consider enabling this flag. You may need to slightly increase the"
                " number of velocity iterations (setting it to 1 or 2 rather than 0), together with this flag, to"
                " improve the accuracy of velocity updates."
            )

        if not cls._cfg.enable_stabilization and cls._cfg.dt > 0.0333:
            cls._sim.logger.warning(
                "Large simulation step size (> 0.0333 seconds) is not recommended without enabling stabilization."
                " Consider setting the `enable_stabilization` flag to True in PhysxManagerCfg, or reducing the"
                " simulation step size if you run into physics issues."
            )

    @classmethod
    def _load_fabric_interface(cls) -> None:
        """Loads the fabric interface if enabled."""
        if cls._sim is None or cls._cfg is None:
            return

        carb_settings = cls._sim.carb_settings

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        fabric_enabled = extension_manager.is_extension_enabled("omni.physx.fabric")

        if cls._cfg.use_fabric:
            if not fabric_enabled:
                extension_manager.set_extension_enabled_immediate("omni.physx.fabric", True)

            # load fabric interface
            from omni.physxfabric import get_physx_fabric_interface

            # acquire fabric interface
            cls._fabric_iface = get_physx_fabric_interface()
            if hasattr(cls._fabric_iface, "force_update"):
                # The update method in the fabric interface only performs an update if a physics step has occurred.
                # However, for rendering, we need to force an update since any element of the scene might have been
                # modified in a reset (which occurs after the physics step) and we want the renderer to be aware of
                # these changes.
                cls._update_fabric = cls._fabric_iface.force_update
            else:
                # Needed for backward compatibility with older Isaac Sim versions
                cls._update_fabric = cls._fabric_iface.update
        else:
            if fabric_enabled:
                extension_manager.set_extension_enabled_immediate("omni.physx.fabric", False)
            # set fabric interface to None
            cls._fabric_iface = None

        # set carb settings for fabric
        carb_settings.set_bool("/isaaclab/physics/fabric_enabled", cls._cfg.use_fabric)
        carb_settings.set_bool("/physics/updateToUsd", not cls._cfg.use_fabric)
        carb_settings.set_bool("/physics/updateParticlesToUsd", not cls._cfg.use_fabric)
        carb_settings.set_bool("/physics/updateVelocitiesToUsd", not cls._cfg.use_fabric)
        carb_settings.set_bool("/physics/updateForceSensorsToUsd", not cls._cfg.use_fabric)
        carb_settings.set_bool("/physics/updateResidualsToUsd", not cls._cfg.use_fabric)
        # disable simulation output window visibility
        carb_settings.set_bool("/physics/visualizationDisplaySimulationOutput", False)
