# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
import logging
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import warp as wp

from .. import sim as sim_utils
from ..app.settings_manager import get_settings_manager
from ..markers.vis_marker_registry import VisMarkerRegistry
from ..physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from ..physics.physics_manager_cfg import _resolve_physx_auto_cfg
from ..renderers.render_context import RenderContext
from ..renderers.renderer_cfg import RendererCfg
from ..scene_data import REQUIRES_STAGE_AND_MODEL, SceneDataProvider
from ..utils import instantiate
from ..utils.string import clear_resolve_matching_names_cache
from ..utils.version import has_kit
from ..visualizers.base_visualizer import BaseVisualizer
from ..visualizers.visualizer_cfg import get_visualizer_install_hint, parse_visualizer_csv, resolve_visualizer_cfgs
from .utils import create_new_stage
from .utils import stage as stage_utils

if TYPE_CHECKING:
    from pxr import Usd

    from ..cloner.clone_plan import ClonePlan

from .simulation_cfg import BackendCfg, SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg

logger = logging.getLogger(__name__)


def _resolve_physics_cfg(physics_cfg: PhysicsCfg | None, use_isaac_sim: bool) -> PhysicsCfg:
    """Resolve a simulation physics config to a concrete backend."""
    if physics_cfg is None:
        # core must not import a backend package at module level
        from isaaclab_physx.physics import PhysxCfg

        physics_cfg = PhysxCfg()
    elif not isinstance(physics_cfg, PhysicsCfg):
        raise TypeError(f"SimulationCfg.physics must be a concrete PhysicsCfg, got {type(physics_cfg).__name__}.")

    return _resolve_physx_auto_cfg(physics_cfg, use_isaac_sim=use_isaac_sim)


class SimulationContext:
    """Controls simulation lifecycle including physics stepping and rendering.

    This singleton class manages:

    * Physics configuration (time-step, solver parameters via :class:`isaaclab.sim.SimulationCfg`)
    * Simulation state (play, pause, step, stop)
    * Rendering and visualization

    Use :meth:`instance` to retrieve the live context. Construction always creates a new context
    and raises if one already exists; call :meth:`clear_instance` before constructing a replacement.
    """

    # SINGLETON PATTERN

    _instance: SimulationContext | None = None
    _reset_callbacks: ClassVar[dict[str, Callable[[SimulationContext], None]]] = {}

    @classmethod
    def instance(cls) -> SimulationContext | None:
        """Get the singleton instance, or None if not created."""
        return cls._instance

    @classmethod
    def add_reset_callback(cls, name: str, fn: Callable[[SimulationContext], None]) -> None:
        """Register a callback to fire after every :meth:`reset` of any simulation context.

        Unlike :meth:`add_render_callback`, the callback is registered on the class, so a launcher
        can install it before the script it runs creates its simulation context.

        Args:
            name: Unique identifier. Silently replaces any existing callback with the same name.
            fn: Callable invoked with the reset simulation context once its visualizers are ready.
        """
        cls._reset_callbacks[name] = fn

    @classmethod
    def remove_reset_callback(cls, name: str) -> None:
        """Unregister a previously registered reset callback.

        Args:
            name: Identifier passed to :meth:`add_reset_callback`. No-op if not found.
        """
        cls._reset_callbacks.pop(name, None)

    def __init__(self, cfg: SimulationCfg | None = None):
        """Initialize the simulation context.

        Args:
            cfg: Simulation configuration. Defaults to None (uses default config).

        Raises:
            RuntimeError: If a simulation context already exists.
        """
        if type(self)._instance is not None:
            raise RuntimeError(
                "A SimulationContext already exists. Use SimulationContext.instance() to retrieve it,"
                " or call SimulationContext.clear_instance() before constructing a replacement."
            )

        from pxr import UsdUtils  # noqa: PLC0415

        # Store config
        self.cfg = SimulationCfg() if cfg is None else cfg
        self._backend_registry: list[tuple[Any, Any]] = []
        self.clone_contexts: dict[type, Any] = {}
        """Clone-context instances registered by type before plan dispatch; not native resource owners."""

        use_isaac_sim = has_kit()
        self._physics = _resolve_physics_cfg(self.cfg.physics, use_isaac_sim=use_isaac_sim)
        self.cfg.physics = self._physics
        self._physics.class_type._prepare_stage_creation()

        # Get or create stage based on config
        stage_cache = UsdUtils.StageCache.Get()
        if self.cfg.create_stage_in_memory:
            self.stage = create_new_stage()
        else:
            # Prefer the thread-local current stage (set by create_new_stage / test fixtures)
            # over cache lookup, since the cache may contain stale stages from prior tests.
            current = getattr(stage_utils._context, "stage", None)
            if current is not None:
                self.stage = current
            else:
                all_stages = stage_cache.GetAllStages() if stage_cache.Size() > 0 else []  # type: ignore[union-attr]
                self.stage = all_stages[0] if all_stages else create_new_stage()

        # Ensure stage is in the USD cache
        stage_id = stage_cache.GetId(self.stage).ToLongInt()  # type: ignore[union-attr]
        if stage_id < 0:
            stage_cache.Insert(self.stage)  # type: ignore[union-attr]

        # Set as current stage in thread-local context for get_current_stage()
        stage_utils._context.stage = self.stage

        if use_isaac_sim:
            from isaaclab_physx.app.kit_stage import KitStageBackendCfg  # noqa: PLC0415

            # Kit extensions (PhysX views, articulations, the viewport) find the stage through Kit's USD context
            self.get_or_create_backend(KitStageBackendCfg(stage=self.stage))

        # Acquire settings interface (SettingsManager: standalone dict or Omniverse when available)
        self.settings = get_settings_manager()
        # Normalize the visualizers to a list, applying the --visualizer selection a launch recorded for the
        # config built afterwards. Without a selection (the setting absent or empty), a config built by the caller
        # keeps the visualizers it lists.
        pending_visualizers = self.get_setting("/isaaclab/visualizer/types")
        max_visible_envs = self.get_setting("/isaaclab/visualizer/max_visible_envs")
        self.cfg.visualizer_cfgs = resolve_visualizer_cfgs(
            self.cfg.visualizer_cfgs,
            parse_visualizer_csv(pending_visualizers) if pending_visualizers else None,
            None if max_visible_envs is None or max_visible_envs < 0 else max_visible_envs,
        )

        # Initialize USD physics scene and physics manager
        self._init_usd_physics_scene()

        # Normalize "cuda" -> "cuda:<id>" now that the USD physics scene is initialized
        # and /physics/cudaDevice is available. Update cfg.device in-place so all
        # downstream code (physics backends, assets, sensors) sees a consistent value.
        if "cuda" in self.cfg.device and ":" not in self.cfg.device:
            cuda_device = self.get_setting("/physics/cudaDevice")
            device_id = max(0, int(cuda_device) if cuda_device is not None else 0)
            self.cfg.device = f"cuda:{device_id}"

        # Select the process device before constructing any physics, rendering, or visualization backend.
        if "cuda" in self.cfg.device:
            torch.cuda.set_device(self.cfg.device)
        wp.set_device(self.cfg.device)

        self.physics_manager: type[PhysicsManager] = self._physics.class_type
        # Must be set before physics_manager.initialize() so that any render callbacks
        # registered during initialize() (e.g. PhysxManager's headless video pump) succeed.
        self._render_callbacks: dict[str, tuple[int, Callable[[Any], None]]] = {}
        self.physics_manager.initialize(self)

        # Construct visualizers before cloning; initialize their runtime bindings after physics is ready.
        self._scene_data_provider = SceneDataProvider(self.physics_manager.get_scene_data_backend())
        self._visualizers: list[BaseVisualizer] = []
        self._pending_visualizers: list[BaseVisualizer] = []
        self._visualizers_started = False
        self._reset_requested: bool = False
        # Set by the visualizers and renderers in use; read by the scene data provider.
        self.requires_usd_stage = False
        self.requires_newton_model = False
        # Clone plan published before cfg-owned scene construction. Constructors and
        # backends therefore consume the same immutable layout through one lifecycle.
        self._clone_plan: ClonePlan | None = None
        # Default visualization dt used before/without visualizer initialization.
        self._viz_dt = self.cfg.dt * self.cfg.render_interval

        # Cache commonly-used settings (these don't change during runtime)
        self._has_gui = bool(self.get_setting("/isaaclab/has_gui"))
        self._has_offscreen_render = bool(self.get_setting("/isaaclab/render/offscreen"))
        self._xr_enabled = bool(self.get_setting("/isaaclab/xr/enabled"))
        # Note: has_rtx_sensors is NOT cached because it changes when Camera sensors are created.
        # It is a global setting flipped to True by RTX Camera creation (see Camera._initialize_impl)
        # and is never flipped back. Reset it here so a fresh SimulationContext reflects its own
        # cameras rather than inheriting a stale True from a previously torn-down simulation. RTX
        # cameras created for this instance re-set it to True before it is read.
        self.set_setting("/isaaclab/render/rtx_sensors", False)
        # Preserve rendering initialization, then avoid continuous Fabric synchronization when
        # the only visualizer is used for on-demand headless capture.
        self.set_setting("/physics/fabricUpdateTransformations", self.is_rendering)
        # Set by camera sensors, which draw visual-only geometry regardless of renderer backend.
        self._visual_shapes_required = False
        self._pending_camera_view: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self.vis_marker_registry = VisMarkerRegistry()

        # Simulation state
        self._is_playing = False
        self._is_stopped = True

        # Monotonic physics-step counter used by camera sensors for data freshness checks.
        self._physics_step_count: int = 0
        # Monotonic render-generation counter. This increments whenever render()
        # is executed and lets downstream camera freshness logic distinguish
        # render/reset transitions that occur without advancing physics steps.
        self._render_generation: int = 0

        # Shared renderers for all Camera sensors (compatible renderer_cfg only).
        self._render_context = RenderContext(self._backend_registry)

        # Run renderer post-physics setup.
        self.physics_manager.register_callback(
            lambda _payload: self._render_context.ensure_initialize(),
            PhysicsEvent.PHYSICS_READY,
            order=5,
        )
        self.physics_manager.register_callback(
            lambda _payload: self._render_context.close(), PhysicsEvent.STOP, order=100
        )

        # Publish the context before configured consumers register their clone requirements.
        type(self)._instance = self
        self._create_visualizers()

    def _init_usd_physics_scene(self) -> None:
        """Create and configure the USD physics scene."""
        from pxr import Gf, UsdGeom, UsdPhysics  # noqa: PLC0415

        cfg = self.cfg
        with sim_utils.use_stage(self.stage):
            # Set stage conventions for metric units
            UsdGeom.SetStageUpAxis(self.stage, "Z")
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)

            # Find and delete any existing physics scene.
            # Collect paths first to avoid mutating the stage while traversing,
            # which can invalidate the USD iterator.
            physics_scene_paths = [
                prim.GetPath().pathString for prim in self.stage.Traverse() if prim.GetTypeName() == "PhysicsScene"
            ]
            for path in physics_scene_paths:
                sim_utils.delete_prim(path, stage=self.stage)

            # Create a new physics scene
            if self.stage.GetPrimAtPath(cfg.physics_prim_path).IsValid():
                raise RuntimeError(f"A prim already exists at path '{cfg.physics_prim_path}'.")

            physics_scene = UsdPhysics.Scene.Define(self.stage, cfg.physics_prim_path)

            # Pre-create gravity tensor to avoid torch heap corruption issues (torch 2.1+)
            gravity = torch.tensor(cfg.gravity, dtype=torch.float32, device=self.cfg.device)
            gravity_magnitude = torch.norm(gravity).item()

            if gravity_magnitude == 0.0:
                gravity_direction = [0.0, 0.0, -1.0]
            else:
                gravity_direction = (gravity / gravity_magnitude).tolist()

            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
            physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)

    @property
    def physics_sim_view(self):
        """Returns the physics simulation view."""
        return self.physics_manager.get_physics_sim_view()

    @property
    def device(self) -> str:
        """Returns the device on which the simulation is running."""
        return self.physics_manager.get_device()

    @property
    def backend(self) -> str:
        """Returns the tensor backend being used ("numpy" or "torch")."""
        return self.physics_manager.get_backend()

    @property
    def has_gui(self) -> bool:
        """Returns whether GUI is enabled (cached at init)."""
        return self._has_gui

    @property
    def has_offscreen_render(self) -> bool:
        """Returns whether offscreen rendering is enabled (cached at init)."""
        return self._has_offscreen_render

    def has_active_visualizers(self) -> bool:
        """Return whether any visualizer path is active for rendering/camera control."""
        return self._has_continuous_visualizers() or bool(self.get_setting("/isaaclab/video/auto_start_kit"))

    def is_running(self) -> bool:
        """Return whether the simulation should keep running.

        Without visualizers it keeps running until the caller stops. Once visualizers were started, it keeps
        running while one of them is still open, so closing the last one ends the loop.
        """
        return not self._visualizers_started or any(viz.is_running() and not viz.is_closed for viz in self._visualizers)

    def require_visual_shapes(self) -> None:
        """Record that something in this simulation draws the physics model's visual-only shapes.

        Camera sensors call this from their constructor, before cloning runs, so backends that
        import visual geometry lazily (see :attr:`isaaclab_newton.physics.NewtonCfg.load_visual_shapes`)
        know the geometry is needed even when no viewer or offscreen capture is active.
        """
        self._visual_shapes_required = True

    @property
    def visual_shapes_required(self) -> bool:
        """Whether :meth:`require_visual_shapes` was called for this simulation."""
        return self._visual_shapes_required

    def can_render_rgb_array(self) -> bool:
        """Return whether rgb-array rendering is currently available, including from a headless visualizer."""
        return (
            self.has_gui or self.has_offscreen_render or self.has_active_visualizers() or bool(self.cfg.visualizer_cfgs)
        )

    @property
    def is_rendering(self) -> bool:
        """Returns whether *continuous* rendering is active (GUI, RTX sensors, visualizers, or XR).

        This drives the per-step render/Kit-pump loop, so it deliberately excludes headless
        offscreen rendering (``--video`` / ``rgb_array``). Offscreen frames are produced on
        demand when a frame is actually requested (via :meth:`render`), not on every step; see
        :meth:`has_offscreen_render` and :meth:`can_render_rgb_array` for the capability checks.
        """
        return (
            self._has_gui
            or self.get_setting("/isaaclab/render/rtx_sensors")
            or self._has_continuous_visualizers()
            or self._xr_enabled
        )

    def get_physics_dt(self) -> float:
        """Returns the physics time step [s]."""
        return self.cfg.dt

    def get_physics_step_count(self) -> int:
        """Return the monotonic physics step counter (incremented each :meth:`step`)."""
        return self._physics_step_count

    @property
    def render_context(self) -> RenderContext:
        """Shared rendering state for camera backends and visual materials."""
        return self._render_context

    @property
    def render_generation(self) -> int:
        """Returns a monotonic counter for render() executions."""
        return self._render_generation

    def _apply_default_visualizer_cfg(self, cfg: Any) -> None:
        """Apply shared default visualizer settings to a backend-specific config.

        Only propagates fields that were **explicitly set** in ``default_visualizer_cfg``
        (i.e. differ from its own class defaults) and are still at the target cfg's
        class defaults. Backend-specific defaults, such as the streaming renderer,
        do not transfer between visualizer types.
        """
        default_cfg = self.cfg.default_visualizer_cfg
        if default_cfg is None:
            return
        source_defaults, target_defaults = type(default_cfg)(), type(cfg)()
        for field in fields(default_cfg):
            if field.name in ("class_type", "visualizer_type") or not hasattr(cfg, field.name):
                continue
            default_val = getattr(default_cfg, field.name)
            if default_val == getattr(source_defaults, field.name):
                continue
            if getattr(cfg, field.name) != getattr(target_defaults, field.name):
                continue
            setattr(cfg, field.name, default_val)

    def resolve_visualizer_types(self) -> list[str]:
        """Return the types of the visualizers in :attr:`SimulationCfg.visualizer_cfgs`."""
        return [cfg.visualizer_type for cfg in self.cfg.visualizer_cfgs if cfg.visualizer_type]

    def _has_continuous_visualizers(self) -> bool:
        """Return whether the configured visualizers require per-step updates."""
        # only the Kit and Newton configs have ``headless``
        return any(cfg.visualizer_type and not getattr(cfg, "headless", False) for cfg in self.cfg.visualizer_cfgs)

    def _resolve_visualizer_cfgs(self) -> list[Any]:
        """Return the configured visualizers with the shared defaults applied, plus a Kit visualizer for XR."""
        resolved = list(self.cfg.visualizer_cfgs)
        for cfg in resolved:
            self._apply_default_visualizer_cfg(cfg)

        # XR auto-start needs a Kit visualizer to publish SDP transforms before pumping the app.
        if (
            self._xr_enabled
            and self.get_setting("/isaaclab/xr/auto_start")
            and not any(cfg.visualizer_type == "kit" for cfg in resolved)
        ):
            try:
                # isaaclab_visualizers is optional
                from isaaclab_visualizers.kit import KitVisualizerCfg
            except ImportError as exc:
                logger.warning(
                    "[SimulationContext] XR mode could not auto-inject a KitVisualizer: %s. %s",
                    exc,
                    get_visualizer_install_hint("kit"),
                )
            else:
                resolved.append(KitVisualizerCfg())
                logger.info("[SimulationContext] Auto-injecting KitVisualizer for XR app-update pumping.")

        return resolved

    def _create_visualizers(self) -> None:
        """Construct cfg-owned consumers and publish their requirements before scene cloning."""
        for cfg in self._resolve_visualizer_cfgs():
            if cfg.visualizer_type is not None:
                requires_stage, requires_model = REQUIRES_STAGE_AND_MODEL[cfg.visualizer_type]
                self.requires_usd_stage |= requires_stage
                self.requires_newton_model |= requires_model
            self._render_context.clone_contexts.update(cfg.cloning_contexts)
            self._pending_visualizers.append(instantiate(cfg))

    def initialize_visualizers(self, config_filter: Callable[[Any], bool] | None = None) -> None:
        """Initialize the constructed visualizers after their shared scene has been cloned.

        Args:
            config_filter: Predicate on a visualizer config selecting which pending visualizers to
                initialize, e.g. only the consumers needed before graph capture. Defaults to None (all).
        """
        for visualizer in tuple(self._pending_visualizers):
            if config_filter is not None and not config_filter(visualizer.cfg):
                continue
            visualizer.initialize(self._scene_data_provider)
            self._pending_visualizers.remove(visualizer)
            self._visualizers.append(visualizer)
            self._visualizers_started = True
            if self._pending_camera_view is not None:
                visualizer.set_camera_view(*self._pending_camera_view)
        if not self._pending_visualizers:
            self._pending_camera_view = None

    def get_scene_data_provider(self) -> SceneDataProvider:
        """Return the scene data provider shared by visualizers and renderers."""
        return self._scene_data_provider

    def register_interactive_scene(self, scene) -> None:
        """Register the active scene so scene data providers can expose scene-owned sensors."""
        self._interactive_scene = scene
        if self._scene_data_provider is not None:
            self._scene_data_provider.set_interactive_scene(scene)

    def get_clone_plan(self) -> ClonePlan | None:
        """Return the clone plan published by the scene.

        Set before cfg-owned scene construction and retained through backend replication.
        ``None`` until a clone lifecycle begins.
        """
        return self._clone_plan

    def set_clone_plan(self, plan: ClonePlan | None) -> None:
        """Set the cloner's active clone plan."""
        self._clone_plan = plan

    @property
    def visualizers(self) -> list[BaseVisualizer]:
        """Returns the list of active visualizers."""
        return self._visualizers

    def get_rendering_dt(self) -> float:
        """Return rendering dt, allowing visualizer-specific override."""
        for viz in self._visualizers:
            viz_dt = viz.get_rendering_dt()
            if viz_dt is not None and viz_dt > 0:
                return float(viz_dt)
        return self._viz_dt

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view on all visualizers that support it."""
        self._pending_camera_view = (tuple(eye), tuple(target))
        for viz in self._visualizers:
            viz.set_camera_view(eye, target)

    def add_render_callback(self, name: str, fn: Callable[[Any], None], order: int = 0) -> None:
        """Register a callback to fire after every render step.

        Args:
            name: Unique identifier. Silently replaces any existing callback with the same name.
            fn: Callable invoked with a single ``None`` argument after each :meth:`render` call.
            order: Execution order relative to other callbacks. Lower values fire first.
        """
        self._render_callbacks[name] = (order, fn)

    def remove_render_callback(self, name: str) -> None:
        """Unregister a previously registered render callback.

        Args:
            name: Identifier passed to :meth:`add_render_callback`. No-op if not found.
        """
        self._render_callbacks.pop(name, None)

    def forward(self) -> None:
        """Update kinematics without stepping physics."""
        self.physics_manager.forward()

    def _prepare_newton_visualizer_for_capture(self, _payload=None) -> None:
        """Initialize or rebind the Newton viewer before solver graph capture."""
        # Picking applies forces inside solver substeps, so its kernels and buffers
        # must exist during graph capture. Render-only viewers can initialize later.
        self.initialize_visualizers(self._requires_pre_capture_newton_init)
        for viz in (viz for viz in self._visualizers if self._requires_pre_capture_newton_init(viz.cfg)):
            viz.reset(soft=False)

    @staticmethod
    def _requires_pre_capture_newton_init(cfg: Any) -> bool:
        """Return whether a config contributes Newton picking inputs to capture."""
        return (
            cfg.visualizer_type in {"newton_gl", "newton_rtx"}
            and bool(getattr(cfg, "enable_picking", False))
            and not bool(getattr(cfg, "headless", False))
        )

    def reset(self, soft: bool = False) -> None:
        """Reset the simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        self.physics_manager.reset(soft)
        for viz in self._visualizers:
            viz.reset(soft)
        # Initialize visualizers not prepared by a backend-specific pre-capture hook.
        self.initialize_visualizers()
        self._render_context.finalize_consumers(self._visualizers, rebuild=not soft)
        # Start the timeline so the play button is pressed
        self.physics_manager.play()
        self._is_playing = True
        self._is_stopped = False
        for callback in tuple(self._reset_callbacks.values()):
            callback(self)

    def step(self, render: bool = True) -> None:
        """Step physics and optionally render.

        If the timeline is paused (e.g. via the GUI), this method blocks and keeps
        the visualizer responsive until the timeline is resumed or stopped.

        Args:
            render: Whether to render the scene after stepping. Defaults to True.
        """
        # Block while the GUI timeline is paused so the entire training loop freezes.
        # See: https://github.com/isaac-sim/IsaacLab/issues/4279
        self.physics_manager.wait_for_playing()
        self._physics_step_count += 1
        self.physics_manager.step()
        if render and self.is_rendering:
            self.render()

    def render(self, mode: int | None = None, skip_app_pumping: bool = False) -> None:
        """Update visualizers and render the scene.

        Calls update_visualizers() so visualizers run at the render cadence (not at
        every physics step). Camera sensors drive their configured renderer when
        fetching data. Physics-backend recording hooks (e.g. Kit/RTX headless video pump) fire through
        :meth:`add_render_callback` so they are not hard-coded in this class.

        **Kit vs. standalone visualizers:**  The Kit app loop (``app.update()``) is the
        only way to drive camera/RTX sensor rendering and viewport GUI updates; it
        cannot be split into "cameras only" and "GUI only".  Standalone visualizers
        (Newton, Rerun, Viser) have self-contained ``step()`` methods that never call
        ``app.update()``, so they can run independently of camera rendering.  The
        ``skip_app_pumping`` flag exploits this distinction: when True, Kit is skipped
        while standalone visualizers continue to update.

        Args:
            mode: Unused. Kept for backward compatibility.
            skip_app_pumping: When True, skip visualizers whose :meth:`~BaseVisualizer.pumps_app_update`
                returns True (e.g. KitVisualizer).  This disables the Kit app loop and camera
                updates while still stepping standalone visualizers (Newton, Rerun, Viser).
                Used by environment ``step()`` when ``render_enabled`` is False.
        """
        self.physics_manager.pre_render()
        self.update_visualizers(self.get_rendering_dt(), skip_app_pumping=skip_app_pumping)
        self.physics_manager.after_visualizers_render()
        for _, callback in sorted(self._render_callbacks.values(), key=lambda x: x[0]):
            callback(None)
        self._render_generation += 1

    def update_visualizers(self, dt: float, skip_app_pumping: bool = False) -> None:
        """Update visualizers without triggering renderer/GUI.

        Args:
            dt: Simulation time-step in seconds.
            skip_app_pumping: When True, skip visualizers whose :meth:`~BaseVisualizer.pumps_app_update`
                returns True (e.g. KitVisualizer). This is used when the environment's ``render_enabled``
                flag is False — cameras and the Kit app loop are skipped, but standalone visualizers
                (Newton, Rerun, Viser) still receive updates.
        """
        if not self._visualizers:
            return

        for viz in self._visualizers:
            viz.flush_startup_messages()

        if self._should_forward_before_visualizer_update():
            self.physics_manager.forward()

        # Marker callbacks update VisualizationMarkers state; visualizer step()
        # consumes that state later in this method. Live-plot panels register in the same
        # registry and their flag is independent of markers, so gate on either capability.
        if any(
            viz.supports_markers() or (viz.supports_live_plots() and viz.cfg.enable_live_plots)
            for viz in self._visualizers
        ):
            self.vis_marker_registry.dispatch_callbacks()

        visualizers_to_remove = []
        for viz in self._visualizers:
            try:
                # When skip_app_pumping is set, skip Kit-like visualizers that call app.update()
                if skip_app_pumping and viz.pumps_app_update():
                    continue
                if viz.is_closed or not viz.is_running():
                    state = "closed" if viz.is_closed else "not running"
                    logger.info("Visualizer %s: %s", state, type(viz).__name__)
                    visualizers_to_remove.append(viz)
                    continue
                if viz.is_rendering_paused():
                    # Keep non-Kit visualizer event loops responsive while rendering is paused.
                    # Newton/Rerun/Viser need step(0.0) so GL/UI can process input (e.g. Resume).
                    # Kit is skipped: step() would call app.update(), which must not run during pause.
                    if not viz.pumps_app_update():
                        viz.step(0.0)
                    continue
                while viz.is_training_paused() and viz.is_running():
                    viz.step(0.0)
                viz.step(dt)
            except Exception as exc:
                logger.error("Error stepping visualizer '%s': %s", type(viz).__name__, exc)
                visualizers_to_remove.append(viz)

        for viz in visualizers_to_remove:
            try:
                viz.close()
                self._visualizers.remove(viz)
                logger.info("Removed visualizer: %s", type(viz).__name__)
            except Exception as exc:
                logger.error("Error closing visualizer: %s", exc)

    def _should_forward_before_visualizer_update(self) -> bool:
        """Return True if any visualizer requires pre-step forward kinematics."""
        return any(viz.requires_forward_before_step() for viz in self._visualizers)

    def play(self) -> None:
        """Start or resume the simulation."""
        self.physics_manager.play()
        for viz in self._visualizers:
            viz.play()
        self._is_playing = True
        self._is_stopped = False

    def pause(self) -> None:
        """Pause the simulation (can be resumed with play)."""
        self.physics_manager.pause()
        for viz in self._visualizers:
            viz.pause()
        self._is_playing = False

    def stop(self) -> None:
        """Stop the simulation completely."""
        self.physics_manager.stop()
        for viz in self._visualizers:
            viz.stop()
        self._is_playing = False
        self._is_stopped = True

    def request_reset(self) -> None:
        """Request an episode reset from a UI control (e.g. the Kit window button).

        The request is consumed on the next call to :meth:`consume_reset_request`.
        """
        self._reset_requested = True

    def consume_reset_request(self) -> bool:
        """Return ``True`` if any visualizer or UI control requested an episode reset and clear the flag.

        Checks both the simulation-context-level flag (set by :meth:`request_reset`) and
        each visualizer's own flag. All flags are cleared atomically so a single reset
        is triggered even when multiple sources fire in the same step.

        Returns:
            ``True`` once when a reset was requested, then ``False`` until the next request.
        """
        requested = self._reset_requested
        self._reset_requested = False
        for viz in self._visualizers:
            requested |= viz.consume_reset_request()
        return requested

    def is_playing(self) -> bool:
        """Returns True if simulation is playing (not paused or stopped)."""
        return self._is_playing

    def is_stopped(self) -> bool:
        """Returns True if simulation is stopped (not just paused)."""
        return self._is_stopped

    def set_setting(self, name: str, value: Any) -> None:
        """Set a setting value."""
        self.settings.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Get a setting value."""
        return self.settings.get(name)

    def get_or_create_backend(self, cfg: Any) -> Any:
        """Return the simulation-owned object for a construction configuration.

        Equal configurations of the same concrete type share a resource. Finalize configurations
        before registration and treat them as read-only afterward; use a new cfg for new settings.
        ``BackendCfg`` declares a resource requiring ``close()``; other cfgs declare Python-owned data.

        Args:
            cfg: Construction inputs. A cache miss constructs ``instantiate(cfg)``.

        Returns:
            The existing or newly constructed resource.
        """
        for registered_cfg, resource in self._backend_registry:
            if type(registered_cfg) is type(cfg) and registered_cfg == cfg:
                return resource
        if isinstance(cfg, RendererCfg):
            self._render_context.validate_renderer_cfg(cfg)
        resource = instantiate(cfg)
        self._backend_registry.append((cfg, resource))
        if isinstance(cfg, RendererCfg):
            self._render_context.register_renderer(cfg, resource)
        return resource

    def close_backend(self, backend: Any) -> None:
        """Release one registered object by identity.

        ``BackendCfg`` resources are closed; plain construction data only loses its registry reference.
        A failed release retains the registry entry so teardown can be retried.

        Args:
            backend: The exact registered resource to close, not its configuration.

        Raises:
            KeyError: The backend is not registered with this context.
        """
        for index, (cfg, resource) in enumerate(self._backend_registry):
            if resource is backend:
                if isinstance(cfg, BackendCfg):
                    resource.close()
                self._backend_registry.pop(index)
                if isinstance(cfg, RendererCfg):
                    self._render_context._prepared_renderer_ids.discard(id(resource))
                return
        raise KeyError(backend)

    @classmethod
    def clear_instance(cls) -> None:
        """Stop the simulation, clean up resources, and clear the singleton instance."""
        instance = cls._instance
        if instance is not None:
            teardown_errors: list[Exception] = []

            def run_cleanup(callback: Callable[[], Any]) -> None:
                try:
                    callback()
                except Exception as exc:
                    teardown_errors.append(exc)

            try:
                # Stop task producers and deliver STOP before releasing their resources.
                run_cleanup(instance.stop)
                # Detach PhysX before any stage-bound resource or prim is deleted.
                run_cleanup(instance.physics_manager.close)

                # Close camera renderers after STOP invalidates camera-owned render data and
                # before the stage is closed so stage-bound renderer resources remain valid.
                run_cleanup(instance._render_context.close)
                for cfg, resource in tuple(instance._backend_registry):
                    if isinstance(cfg, RendererCfg):
                        run_cleanup(lambda resource=resource: instance.close_backend(resource))

                # Give every visualizer a chance to release its resources.
                for viz in (*instance._visualizers, *instance._pending_visualizers):
                    run_cleanup(viz.close)
                instance._visualizers.clear()
                instance._pending_visualizers.clear()

                instance.clone_contexts.clear()
                # Newest first: the Kit USD-context backend, registered first, closes last but
                # before close_stage() clears the stage cache.
                for cfg, resource in reversed(instance._backend_registry):
                    if isinstance(cfg, BackendCfg) and not isinstance(cfg, RendererCfg):
                        run_cleanup(lambda resource=resource: resource.close())
                instance._backend_registry.clear()

                # Tear down the stage. We skip clear_stage() (prim-by-prim deletion) since
                # close_stage() + app shutdown destroy the entire stage at once.
                run_cleanup(stage_utils.close_stage)

                # Discard cached name-resolution data from destroyed assets.
                run_cleanup(clear_resolve_matching_names_cache)
            finally:
                cls._instance = None
                del instance

            run_cleanup(gc.collect)

            logger.info("SimulationContext cleared")

            if len(teardown_errors) == 1:
                raise teardown_errors[0]
            if teardown_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in teardown_errors)
                msg = (
                    f"SimulationContext.clear_instance(): {len(teardown_errors)} error(s) occurred during teardown:"
                    f" {details}"
                )
                raise RuntimeError(msg) from teardown_errors[0]

    @classmethod
    def clear_stage(cls) -> None:
        """Clear the current USD stage (preserving /World and PhysicsScene).

        Uses a predicate that preserves /World and PhysicsScene while also
        respecting the default deletability checks (ancestral prims, etc.).
        """
        if cls._instance is None:
            return

        def _predicate(prim: Usd.Prim) -> bool:
            return prim.GetPath().pathString != "/World" and prim.GetTypeName() != "PhysicsScene"

        sim_utils.clear_stage(predicate=_predicate)


@contextmanager
def build_simulation_context(
    create_new_stage: bool = True,
    gravity_enabled: bool = True,
    device: str | None = None,
    dt: float = 0.01,
    sim_cfg: SimulationCfg | None = None,
    add_ground_plane: bool = False,
    add_lighting: bool = False,
    auto_add_lighting: bool = False,
) -> Iterator[SimulationContext]:
    """Context manager to build a simulation context with the provided settings.

    Args:
        create_new_stage: Whether to create a new stage. Defaults to True.
        gravity_enabled: Whether to enable gravity. Defaults to True.
        device: Device to run the simulation on. When given alongside ``sim_cfg``,
            overrides ``sim_cfg.device`` so the caller's explicit choice wins
            (most test callers pass both, expecting this behavior). Defaults to
            ``None``, meaning ``sim_cfg.device`` is left untouched and a freshly
            built ``sim_cfg`` uses :class:`SimulationCfg`'s default device.
        dt: Time step for the simulation. Defaults to 0.01.
        sim_cfg: SimulationCfg to use. Defaults to None.
        add_ground_plane: Whether to add a ground plane. Defaults to False.
        add_lighting: Whether to add a dome light. Defaults to False.
        auto_add_lighting: Whether to auto-add lighting if GUI present. Defaults to False.

    Yields:
        The simulation context to use for the simulation.
    """
    sim: SimulationContext | None = None
    try:
        if create_new_stage:
            # ``create_new_stage`` is shadowed here by the bool parameter, so call via the namespace.
            sim_utils.create_new_stage()

        if sim_cfg is None:
            gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
            sim_cfg = SimulationCfg(dt=dt, gravity=gravity)
        if device is not None:
            # Honor the explicit device kwarg in both branches: when sim_cfg is
            # freshly built, this picks the device; when sim_cfg is passed in,
            # this overrides its (possibly default) device. Without the override,
            # callers passing both ``sim_cfg=<built-with-default-device>`` and
            # ``device=cuda:N`` silently got sim_cfg's device, causing warp
            # kernel-launch mismatches when test fixtures allocated tensors on
            # the requested device while assets resolved their device from the
            # untouched sim_cfg.
            sim_cfg.device = device

        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim.has_gui):
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=10000
            )
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        if sim is not None:
            sim.clear_instance()
