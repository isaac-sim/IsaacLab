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
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import torch

import carb
import omni.kit.app
import omni.physics.tensors
import omni.physx
import omni.timeline
import omni.usd
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.utils.string import to_camel_case

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

    from .physx_cfg import PhysxCfg

__all__ = ["IsaacEvents", "PhysxManager"]

logger = logging.getLogger(__name__)


class IsaacEvents(Enum):
    """Events dispatched during simulation lifecycle.

    Note: This enum is kept for backward compatibility. New code should use
    PhysicsEvent from physics_manager for cross-backend compatibility.
    """

    PHYSICS_WARMUP = "isaac.physics_warmup"
    SIMULATION_VIEW_CREATED = "isaac.simulation_view_created"
    PHYSICS_READY = "isaac.physics_ready"
    POST_RESET = "isaac.post_reset"
    PRIM_DELETION = "isaac.prim_deletion"
    PRE_PHYSICS_STEP = "isaac.pre_physics_step"
    POST_PHYSICS_STEP = "isaac.post_physics_step"
    TIMELINE_STOP = "isaac.timeline_stop"


_PHYSICS_EVENT_TO_ISAAC_EVENT: dict[PhysicsEvent, IsaacEvents] = {
    PhysicsEvent.MODEL_INIT: IsaacEvents.PHYSICS_WARMUP,
    PhysicsEvent.PHYSICS_READY: IsaacEvents.PHYSICS_READY,
    PhysicsEvent.STOP: IsaacEvents.TIMELINE_STOP,
}


class AnimationRecorder:
    """Handles animation recording using PhysX PVD interface."""

    def __init__(self, sim_context: SimulationContext):
        self._sim = sim_context
        self._enabled = bool(sim_context.get_setting("/isaaclab/anim_recording/enabled"))
        self._started_at: float | None = None
        self._physx_pvd = None

        if self._enabled:
            self._start_time = sim_context.get_setting("/isaaclab/anim_recording/start_time")
            self._stop_time = sim_context.get_setting("/isaaclab/anim_recording/stop_time")
            self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """Initialize recording directory and PVD interface."""
        from omni.physxpvd.bindings import _physxPvd

        repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self._output_dir = os.path.join(repo_path, "anim_recordings", timestamp).replace("\\", "/").rstrip("/") + "/"
        os.makedirs(self._output_dir, exist_ok=True)

        self._physx_pvd = _physxPvd.acquire_physx_pvd_interface()
        self._sim.set_setting("/persistent/physics/omniPvdOvdRecordingDirectory", self._output_dir)
        self._sim.set_setting("/physics/omniPvdOutputEnabled", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self) -> bool:
        """Update recording state. Returns True if recording finished."""
        if not self._enabled:
            return False
        if self._started_at is None:
            self._started_at = time.time()
        if time.time() - self._started_at > self._stop_time:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        """Finalize and export the recording."""
        logger.warning("[AnimationRecorder] Finishing recording. This may take a few minutes.")

        physx = omni.physx.get_physx_simulation_interface()
        physx.detach_stage()

        stage_path = os.path.join(self._output_dir, "stage_simulation.usdc")
        sim_utils.save_stage(stage_path, save_and_reload_in_place=False)

        ovd_files = [f for f in glob.glob(os.path.join(self._output_dir, "*.ovd")) if not f.endswith("tmp.ovd")]
        if ovd_files and self._physx_pvd:
            input_ovd = max(ovd_files, key=os.path.getctime)
            self._physx_pvd.ovd_to_usd_over_with_layer_creation(
                input_ovd,
                stage_path,
                self._output_dir,
                "baked_animation_recording.usda",
                self._start_time,
                self._stop_time,
                True,
                False,
            )
            self._update_usda_start_time(os.path.join(self._output_dir, "baked_animation_recording.usda"))

        self._sim.set_setting("/physics/omniPvdOutputEnabled", False)

    def _update_usda_start_time(self, file_path: str) -> None:
        """Patch the start time in the exported USDA file."""
        with open(file_path) as f:
            content = f.read()
        match = re.search(r"timeCodesPerSecond\s*=\s*(\d+)", content)
        if match:
            fps = int(match.group(1))
            new_start = int(self._start_time * fps)
            content = re.sub(r"startTimeCode\s*=\s*\d+", f"startTimeCode = {new_start}", content)
            with open(file_path, "w") as f:
                f.write(content)


class PhysxManager(PhysicsManager):
    """Manages PhysX physics simulation lifecycle.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _cfg: ClassVar[PhysxCfg | None] = None

    _timeline: ClassVar[omni.timeline.ITimeline] = omni.timeline.get_timeline_interface()
    _event_bus: ClassVar[carb.eventdispatcher.IEventDispatcher] = carb.eventdispatcher.get_eventdispatcher()
    _physx: ClassVar[omni.physx.IPhysx] = omni.physx.get_physx_interface()
    _physx_sim: ClassVar[omni.physx.IPhysxSimulation] = omni.physx.get_physx_simulation_interface()

    _view: ClassVar[omni.physics.tensors.SimulationView | None] = None
    _view_warp: ClassVar[omni.physics.tensors.SimulationView | None] = None
    _warmup_needed: ClassVar[bool] = True
    _view_created: ClassVar[bool] = False
    _assets_loaded: ClassVar[bool] = True
    _stage_id: ClassVar[int] = -1
    _subscriptions: ClassVar[dict[str, Any]] = {}
    _fabric: ClassVar[Any] = None
    _update_fabric: ClassVar[Callable[[float, float], None] | None] = None
    _anim_recorder: ClassVar[AnimationRecorder | None] = None
    _callback_exception: ClassVar[Exception | None] = None

    class _SimManagerStub:
        """No-op stub for Isaac Sim APIs expecting simulation_manager_interface."""

        def reset(self) -> None:
            pass

        def get_simulation_time(self) -> float:
            return omni.physx.get_physx_interface().get_simulation_time()

        def is_simulating(self) -> bool:
            return omni.physx.get_physx_interface().is_simulating()

        def __getattr__(self, name: str) -> Callable[..., Any]:
            return lambda *a, **kw: None

    # field stubs for Isaac Sim APIs expecting simulation_manager_interface
    _simulation_manager_interface: ClassVar[_SimManagerStub] = _SimManagerStub()
    _physics_scene_apis: ClassVar[dict[str, Any]] = {}
    _message_bus = _event_bus

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager."""
        from isaaclab.sim.utils.stage import get_current_stage_id

        super().initialize(sim_context)
        cls._stage_id = get_current_stage_id()

        cls._setup_subscriptions()
        cls._configure_physics()
        cls._load_fabric()
        cls._anim_recorder = AnimationRecorder(sim_context)

        # force update cycle to apply dt
        sim = PhysicsManager._sim
        sim.set_setting("/app/player/playSimulations", False)  # type: ignore[union-attr]
        omni.kit.app.get_app().update()
        sim.set_setting("/app/player/playSimulations", True)  # type: ignore[union-attr]

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset the physics simulation."""
        if not soft:
            # Ensure views are created (warmup only happens once per stage)
            if cls._view is None:
                cls._warmup_and_create_views()
            # Always dispatch PHYSICS_READY on hard reset to initialize newly registered sensors
            cls._event_bus.dispatch_event(IsaacEvents.PHYSICS_READY.value, payload={})

        device = PhysicsManager._device
        if "cuda" in device:
            torch.cuda.set_device(device)

        if cls._view is not None:
            cls._view._backend.initialize_kinematic_bodies()

        cls.raise_callback_exception_if_any()

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics and fabric for rendering."""
        sim = PhysicsManager._sim
        if cls._fabric is not None and cls._update_fabric is not None:
            if cls._view is not None and sim is not None and sim.is_playing():
                cls._view.update_articulations_kinematic()
            cls._update_fabric(0.0, 0.0)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None:
            return

        if cls._anim_recorder and cls._anim_recorder.enabled and cls._anim_recorder.update():
            logger.warning("Animation recording finished. Shutting down.")
            omni.kit.app.get_app().shutdown()
            return

        cls._physx_sim.simulate(sim.cfg.dt, 0.0)
        cls._physx_sim.fetch_results()

        device = PhysicsManager._device
        if "cuda" in device:
            torch.cuda.set_device(device)

        cls.raise_callback_exception_if_any()

    @classmethod
    def play(cls) -> None:
        """Start or resume the timeline."""
        cls._timeline.play()
        # Pump events so timeline callbacks fire synchronously
        omni.kit.app.get_app().update()

    @classmethod
    def pause(cls) -> None:
        """Pause the timeline."""
        cls._timeline.pause()
        # Pump events so timeline callbacks fire synchronously
        omni.kit.app.get_app().update()

    @classmethod
    def stop(cls) -> None:
        """Stop the timeline."""
        cls._timeline.stop()
        # Pump events so timeline callbacks fire synchronously
        omni.kit.app.get_app().update()

    @classmethod
    def close(cls) -> None:
        """Clean up physics resources."""
        # Detach PhysX from the stage FIRST to prevent shape/actor cleanup errors
        # This disconnects PhysX from USD before any deletion events are fired
        if cls._physx_sim is not None:
            cls._physx_sim.detach_stage()
            # Pump the app to flush pending PhysX cleanup operations
            omni.kit.app.get_app().update()

        # Now invalidate views (they're already disconnected from PhysX)
        cls._invalidate_views()
        cls._subscriptions.clear()

        # Notify listeners that prims are being deleted (safe now since PhysX is detached)
        cls._event_bus.dispatch_event(IsaacEvents.PRIM_DELETION.value, payload={"prim_path": "/"})

        cls._fabric = None
        cls._update_fabric = None
        cls._anim_recorder = None
        cls._warmup_needed = True
        cls._view_created = False
        cls._assets_loaded = True
        cls._callback_exception = None

        super().close()

    @classmethod
    def get_physics_sim_view(cls) -> omni.physics.tensors.SimulationView | None:
        return cls._view

    @classmethod
    def get_physics_sim_device(cls) -> str:
        """Get the physics simulation device (Isaac Sim compatibility alias)."""
        return PhysicsManager.get_device()

    @classmethod
    def assets_loading(cls) -> bool:
        return not cls._assets_loaded

    @classmethod
    def store_callback_exception(cls, exception: Exception) -> None:
        """Store an exception from a callback to be raised later.

        Omniverse event systems catch exceptions internally. Use this to store
        exceptions that should be surfaced after the event dispatch completes.
        """
        if cls._callback_exception is None:
            cls._callback_exception = exception

    @classmethod
    def raise_callback_exception_if_any(cls) -> None:
        """Raise any stored callback exception and clear it.

        Call this after operations that may trigger callbacks (reset, step, etc.)
        to propagate exceptions from Omniverse event callbacks.
        """
        if cls._callback_exception is not None:
            exc = cls._callback_exception
            cls._callback_exception = None
            raise exc

    @classmethod
    def register_callback(
        cls,
        callback: Callable,
        event: PhysicsEvent | IsaacEvents,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback. Accepts both PhysicsEvent and IsaacEvents."""
        if isinstance(event, IsaacEvents):
            cid = cls._callback_id
            cls._callback_id += 1
            cb = cls._wrap_weak_ref(callback) if wrap_weak_ref else callback
            sub = cls._subscribe_isaac(cb, event, order, name)
            cls._callbacks[cid] = (event, cb, order, name, sub)
            return CallbackHandle(cid, cls)
        return super().register_callback(callback, event, order, name, wrap_weak_ref)

    @classmethod
    def _subscribe_to_event(
        cls, callback_id: int, callback: Callable, event: PhysicsEvent, order: int, name: str | None
    ) -> Any:
        """Subscribe to PhysX events. Maps PhysicsEvent → IsaacEvents."""
        isaac_event = _PHYSICS_EVENT_TO_ISAAC_EVENT.get(event)
        return cls._subscribe_isaac(callback, isaac_event, order, name) if isaac_event else None

    @classmethod
    def _unsubscribe_from_event(cls, callback_id: int, event: PhysicsEvent | IsaacEvents, subscription: Any) -> None:
        """Unsubscribe from PhysX/Isaac events."""
        if subscription is not None and hasattr(subscription, "unsubscribe"):
            subscription.unsubscribe()

    @classmethod
    def _subscribe_isaac(cls, callback: Callable, event: IsaacEvents, order: int, name: str | None) -> Any:
        """Subscribe to an IsaacEvents event."""

        def guarded(cb: Callable) -> Callable:
            def wrapper(dt: float) -> Any:
                return cb(dt) if cls._view_created else None

            return wrapper

        if event in (
            IsaacEvents.PHYSICS_WARMUP,
            IsaacEvents.PHYSICS_READY,
            IsaacEvents.POST_RESET,
            IsaacEvents.SIMULATION_VIEW_CREATED,
            IsaacEvents.PRIM_DELETION,
        ):
            return cls._event_bus.observe_event(event_name=event.value, order=order, on_event=callback)
        elif event == IsaacEvents.POST_PHYSICS_STEP:
            return cls._physx.subscribe_physics_on_step_events(guarded(callback), pre_step=False, order=order)
        elif event == IsaacEvents.PRE_PHYSICS_STEP:
            return cls._physx.subscribe_physics_on_step_events(guarded(callback), pre_step=True, order=order)
        elif event == IsaacEvents.TIMELINE_STOP:
            return cls._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), callback, order=order, name=name
            )
        return None

    @classmethod
    def _setup_subscriptions(cls) -> None:
        """Subscribe to timeline events."""
        if "play" in cls._subscriptions:
            return
        stream = cls._timeline.get_timeline_event_stream()
        cls._subscriptions["play"] = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), cls._on_play
        )
        cls._subscriptions["stop"] = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), cls._on_stop
        )
        if "stage_open" not in cls._subscriptions:
            ctx = omni.usd.get_context()
            cls._subscriptions["stage_open"] = cls._event_bus.observe_event(
                event_name=ctx.stage_event_name(omni.usd.StageEventType.OPENED), on_event=cls._on_stage_open
            )

    @classmethod
    def _configure_physics(cls) -> None:
        """Apply all physics settings."""
        # Access base class variables since that's where initialize() sets them
        sim = PhysicsManager._sim
        cfg = PhysicsManager._cfg
        if sim is None or cfg is None:
            return

        device = sim.device

        # global settings (via SettingsManager)
        sim.set_setting("/persistent/omnihydra/useSceneGraphInstancing", True)  # type: ignore[union-attr]
        sim.set_setting("/physics/physxDispatcher", True)  # type: ignore[union-attr]
        sim.set_setting("/physics/disableContactProcessing", True)  # type: ignore[union-attr]
        sim.set_setting("/physics/collisionConeCustomGeometry", False)  # type: ignore[union-attr]
        sim.set_setting("/physics/collisionCylinderCustomGeometry", False)  # type: ignore[union-attr]
        sim.set_setting("/physics/autoPopupSimulationOutputWindow", False)  # type: ignore[union-attr]

        # device setup (set on PhysicsManager so PhysicsManager.get_device() works)
        is_gpu = "cuda" in device
        if is_gpu:
            parts = device.split(":")
            cuda_device = sim.get_setting("/physics/cudaDevice")  # type: ignore[union-attr]
            device_id = int(parts[1]) if len(parts) > 1 else max(0, int(cuda_device) if cuda_device is not None else 0)
            sim.set_setting("/physics/cudaDevice", device_id)  # type: ignore[union-attr]
            sim.set_setting("/physics/suppressReadback", True)  # type: ignore[union-attr]
            PhysicsManager._device = f"cuda:{device_id}"
        else:
            sim.set_setting("/physics/cudaDevice", -1)  # type: ignore[union-attr]
            sim.set_setting("/physics/suppressReadback", False)  # type: ignore[union-attr]
            PhysicsManager._device = "cpu"

        # physx scene api (use sim.cfg for shared parameters like physics_prim_path, dt, physics_material)
        # apply schema and set attributes by name
        sim_cfg = sim.cfg
        stage = sim.stage
        scene_prim = stage.GetPrimAtPath(sim_cfg.physics_prim_path)
        if "PhysxSceneAPI" not in scene_prim.GetAppliedSchemas():
            scene_prim.AddAppliedSchema("PhysxSceneAPI")

        # timestep and frame rate
        steps_per_sec = int(1.0 / sim_cfg.dt)
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:timeStepsPerSecond", steps_per_sec, camel_case=False
        )
        render_interval = max(sim_cfg.render_interval, 1)
        sim.set_setting("/persistent/simulation/minFrameRate", steps_per_sec // render_interval)  # type: ignore[union-attr]

        # gpu dynamics
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:broadphaseType", "GPU" if is_gpu else "MBP", camel_case=False
        )
        sim_utils.safe_set_attribute_on_usd_prim(scene_prim, "physxScene:enableGPUDynamics", is_gpu, camel_case=False)

        # ccd (not supported on gpu)
        enable_ccd = cfg.enable_ccd and not is_gpu
        if cfg.enable_ccd and is_gpu:
            logger.warning("CCD disabled when GPU dynamics is enabled.")
        sim_utils.safe_set_attribute_on_usd_prim(scene_prim, "physxScene:enableCCD", enable_ccd, camel_case=False)

        # solver
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:solverType", "TGS" if cfg.solver_type == 1 else "PGS", camel_case=False
        )
        scene_prim.CreateAttribute("physxScene:solveArticulationContactLast", Sdf.ValueTypeNames.Bool).Set(
            cfg.solve_articulation_contact_last
        )

        # apply remaining cfg attributes to scene (physxScene:*)
        skip = {
            "solver_type",
            "enable_ccd",
            "solve_articulation_contact_last",
            "dt",
            "device",
            "render_interval",
            "gravity",
            "physics_prim_path",
            "use_fabric",
            "physics_material",
            "class_type",
        }
        for key, value in cfg.to_dict().items():  # type: ignore
            if key not in skip:
                attr_name = "bounce_threshold" if key == "bounce_threshold_velocity" else key
                sim_utils.safe_set_attribute_on_usd_prim(
                    scene_prim,
                    f"physxScene:{to_camel_case(attr_name, 'cC')}",
                    value,
                    camel_case=False,
                )

        # default physics material (from SimulationCfg, or create default if None)
        physics_material = sim_cfg.physics_material
        if physics_material is None:
            from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

            physics_material = RigidBodyMaterialCfg()
        mat_path = f"{sim_cfg.physics_prim_path}/defaultMaterial"
        physics_material.func(mat_path, physics_material)
        sim_utils.bind_physics_material(sim_cfg.physics_prim_path, mat_path)

        # warnings
        if cfg.solver_type == 1 and not cfg.enable_external_forces_every_iteration:
            logger.warning("TGS solver with enable_external_forces_every_iteration=False may cause noisy velocities.")
        if not cfg.enable_stabilization and sim_cfg.dt > 0.0333:
            logger.warning("Large timestep without stabilization may cause physics issues.")

    @classmethod
    def _load_fabric(cls) -> None:
        """Load fabric interface if enabled."""
        sim = PhysicsManager._sim
        cfg = PhysicsManager._cfg
        if sim is None or cfg is None:
            return

        use_fabric = sim.cfg.use_fabric
        ext_mgr = omni.kit.app.get_app().get_extension_manager()

        # enable/disable fabric extension
        if use_fabric:
            if not ext_mgr.is_extension_enabled("omni.physx.fabric"):
                ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", True)
            from omni.physxfabric import get_physx_fabric_interface

            cls._fabric = get_physx_fabric_interface()
            cls._update_fabric = getattr(cls._fabric, "force_update", cls._fabric.update)
        else:
            if ext_mgr.is_extension_enabled("omni.physx.fabric"):
                ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", False)
            cls._fabric = None
            cls._update_fabric = None

        # disable usd sync when fabric is enabled (via SettingsManager)
        for key in [
            "updateToUsd",
            "updateParticlesToUsd",
            "updateVelocitiesToUsd",
            "updateForceSensorsToUsd",
            "updateResidualsToUsd",
        ]:
            sim.set_setting(f"/physics/{key}", not use_fabric)  # type: ignore[union-attr]
        sim.set_setting("/isaaclab/fabric_enabled", use_fabric)  # type: ignore[union-attr]
        sim.set_setting("/physics/visualizationDisplaySimulationOutput", False)  # type: ignore[union-attr]

    @classmethod
    def _warmup_and_create_views(cls) -> None:
        """Warm-start physics and create simulation views."""
        if not cls._warmup_needed:
            return

        # Get stage ID first (needed for both warmup and view creation)
        from isaaclab.sim.utils.stage import get_current_stage_id

        stage_id = get_current_stage_id()

        is_gpu = "cuda" in PhysicsManager.get_device()

        # Attach stage to PhysX BEFORE loading/starting - only needed for GPU pipeline.
        # For CPU, the old SimulationManager never called attach_stage() explicitly.
        # Calling attach_stage() + force_load_physics_from_usd() together causes a
        # double-initialization that corrupts the CPU broadphase (MBP) collision setup,
        # causing objects to fall through surfaces non-deterministically.
        if is_gpu:
            cls._physx_sim.attach_stage(stage_id)

        # warmup physx
        cls._physx.force_load_physics_from_usd()
        cls._physx.start_simulation()
        cls._physx.update_simulation(cls.get_physics_dt(), 0.0)
        cls._physx_sim.fetch_results()
        cls._event_bus.dispatch_event(IsaacEvents.PHYSICS_WARMUP.value, payload={})
        cls._warmup_needed = False

        if cls._view_created:
            return

        # Create tensor views
        cls._view = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
        cls._view_warp = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)

        if cls._view:
            cls._view.set_subspace_roots("/")
        if cls._view_warp:
            cls._view_warp.set_subspace_roots("/")

        # Final update after view creation
        cls._physx.update_simulation(cls.get_physics_dt(), 0.0)
        cls._view_created = True

        cls._event_bus.dispatch_event(IsaacEvents.SIMULATION_VIEW_CREATED.value, payload={})
        cls._event_bus.dispatch_event(IsaacEvents.PHYSICS_READY.value, payload={})

    @classmethod
    def _invalidate_views(cls) -> None:
        """Invalidate and clear simulation views."""
        for view in (cls._view, cls._view_warp):
            if view:
                view.invalidate()
        cls._view = None
        cls._view_warp = None
        cls._view_created = False

    @classmethod
    def _on_play(cls, event: Any) -> None:
        sim = PhysicsManager._sim
        if sim is not None and sim.get_setting("/app/player/playSimulations"):  # type: ignore[union-attr]
            cls._warmup_and_create_views()

    @classmethod
    def _on_stop(cls, event: Any) -> None:
        cls._warmup_needed = True
        cls._invalidate_views()

    @classmethod
    def _on_stage_open(cls, event: Any) -> None:
        from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id

        # Guard against stage open events when stage is not yet valid
        stage = get_current_stage()
        if stage is None or not stage.GetRootLayer():
            return

        try:
            new_stage_id = get_current_stage_id()
        except Exception:
            # Stage may not be ready for caching yet
            return

        if new_stage_id == cls._stage_id:
            return

        cls._stage_id = new_stage_id
        cls._callbacks.clear()
        cls._assets_loaded = True

        def on_loading(e: Any) -> None:
            cls._assets_loaded = False

        def on_loaded(e: Any) -> None:
            cls._assets_loaded = True

        ctx = omni.usd.get_context()
        cls._subscriptions["assets_loading"] = cls._event_bus.observe_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADING), on_event=on_loading
        )
        cls._subscriptions["assets_loaded"] = cls._event_bus.observe_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED), on_event=on_loaded
        )
