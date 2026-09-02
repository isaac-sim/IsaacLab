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
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import torch

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.stage as stage_utils
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from isaaclab.physics.physics_manager_cfg import _resolve_physx_auto_cfg
from isaaclab.renderers.render_context import RenderContext
from isaaclab.scene_data import REQUIRES_STAGE_AND_MODEL, SceneDataProvider
from isaaclab.sim.utils import create_new_stage
from isaaclab.utils.string import clear_resolve_matching_names_cache
from isaaclab.utils.version import has_kit
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import _get_visualizer_install_hint

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.cloner.clone_plan import ClonePlan

from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg

logger = logging.getLogger(__name__)


_BackendT = TypeVar("_BackendT")

# Visualizer type names (CLI and config). App launcher parses CSV and stores as a space-separated setting.
_VISUALIZER_TYPES = ("newton_gl", "newton_rtx", "rerun", "viser", "kit")
# Deprecated aliases mapped to their canonical names.
_VISUALIZER_ALIASES = {"newton": "newton_gl"}


def _resolve_physics_cfg(physics_cfg: PhysicsCfg | None, use_isaac_sim: bool) -> PhysicsCfg:
    """Resolve a simulation physics config to a concrete backend."""
    if physics_cfg is None:
        from isaaclab_physx.physics import PhysxCfg

        physics_cfg = PhysxCfg()
    elif not isinstance(physics_cfg, PhysicsCfg):
        raise TypeError(f"SimulationCfg.physics must be a concrete PhysicsCfg, got {type(physics_cfg).__name__}.")

    return _resolve_physx_auto_cfg(physics_cfg, use_isaac_sim=use_isaac_sim)


class SettingsHelper:
    """Helper for typed settings access via SettingsManager."""

    def __init__(self, settings: SettingsManager):
        self._settings = settings

    def set(self, name: str, value: Any) -> None:
        """Set a setting with automatic type routing."""
        if isinstance(value, bool):
            self._settings.set_bool(name, value)
        elif isinstance(value, int):
            self._settings.set_int(name, value)
        elif isinstance(value, float):
            self._settings.set_float(name, value)
        elif isinstance(value, str):
            self._settings.set_string(name, value)
        elif isinstance(value, (list, tuple)):
            self._settings.set(name, value)
        else:
            raise ValueError(f"Unsupported value type for setting '{name}': {type(value)}")

    def get(self, name: str) -> Any:
        """Get a setting value."""
        return self._settings.get(name)


class SimulationContext:
    """Controls simulation lifecycle including physics stepping and rendering.

    This singleton class manages:

    * Physics configuration (time-step, solver parameters via :class:`isaaclab.sim.SimulationCfg`)
    * Simulation state (play, pause, step, stop)
    * Rendering and visualization

    The singleton instance can be accessed using the ``instance()`` class method.
    """

    # SINGLETON PATTERN

    _instance: SimulationContext | None = None

    def __new__(cls, cfg: SimulationCfg | None = None):
        """Enforce singleton pattern."""
        if cls._instance is not None:
            return cls._instance
        return super().__new__(cls)

    @classmethod
    def instance(cls) -> SimulationContext | None:
        """Get the singleton instance, or None if not created."""
        return cls._instance

    def __init__(self, cfg: SimulationCfg | None = None):
        """Initialize the simulation context.

        Args:
            cfg: Simulation configuration. Defaults to None (uses default config).
        """
        if type(self)._instance is not None:
            return  # Already initialized

        from pxr import UsdUtils  # noqa: PLC0415

        # Store config
        self.cfg = SimulationCfg() if cfg is None else cfg
        self._backend_registry: dict[type[object], object] = {}
        self._backend_clone_roles: dict[type[object], set[str]] = {}

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

        # When Kit is running, attach the stage to Kit's USD context so that
        # Kit extensions (PhysX views, Articulation, viewport) can discover it.
        if use_isaac_sim:
            import omni.usd

            kit_context = omni.usd.get_context()
            if kit_context is not None and kit_context.get_stage() is not self.stage:
                kit_context.attach_stage_with_callback(stage_cache.GetId(self.stage).ToLongInt())

        # Acquire settings interface (SettingsManager: standalone dict or Omniverse when available)
        self.settings = SettingsManager.instance()
        self._settings_helper = SettingsHelper(self.settings)

        # Initialize USD physics scene and physics manager
        self._init_usd_physics_scene()

        # Normalize "cuda" -> "cuda:<id>" now that the USD physics scene is initialized
        # and /physics/cudaDevice is available. Update cfg.device in-place so all
        # downstream code (physics backends, assets, sensors) sees a consistent value.
        if "cuda" in self.cfg.device and ":" not in self.cfg.device:
            cuda_device = self.get_setting("/physics/cudaDevice")
            device_id = max(0, int(cuda_device) if cuda_device is not None else 0)
            self.cfg.device = f"cuda:{device_id}"

        self.physics_manager: type[PhysicsManager] = self._physics.class_type
        # Must be set before physics_manager.initialize() so that any render callbacks
        # registered during initialize() (e.g. PhysxManager's headless video pump) succeed.
        self._render_callbacks: dict[str, tuple[int, Callable[[Any], None]]] = {}
        self.physics_manager.initialize(self)

        # Initialize visualizer state (visualizers are created lazily during initialize_visualizers()).
        self._scene_data_provider = SceneDataProvider(self.physics_manager.get_scene_data_backend())
        self._visualizers: list[BaseVisualizer] = []
        self._pending_visualizer_cfgs: list[Any] | None = None
        self._reset_requested: bool = False
        # Set by the visualizers and renderers in use; read by the scene data provider.
        self.requires_usd_stage = False
        self.requires_newton_model = False
        # Clone plan published by InteractiveScene after cloning. Providers (e.g. the
        # Newton visualizer model rebuilder on a PhysX backend) consume this to derive
        # their own backend args. None until a replication session publishes a plan.
        self._clone_plan: ClonePlan | None = None
        # Default visualization dt used before/without visualizer initialization.
        physics_dt = getattr(self.cfg.physics, "dt", None)
        self._viz_dt = (physics_dt if physics_dt is not None else self.cfg.dt) * self.cfg.render_interval

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
        # Set by camera sensors, which draw visual-only geometry regardless of renderer backend.
        self._visual_shapes_required = False
        self._pending_camera_view: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self.vis_marker_registry = VisMarkerRegistry()

        # Simulation state
        self._is_playing = False
        self._is_stopped = True

        # Monotonic physics-step counter used by camera sensors for
        self._physics_step_count: int = 0
        # Monotonic render-generation counter. This increments whenever render()
        # is executed and lets downstream camera freshness logic distinguish
        # render/reset transitions that occur without advancing physics steps.
        self._render_generation: int = 0

        # Shared renderers for all Camera sensors (compatible renderer_cfg only).
        self._render_context = RenderContext()

        # Run renderer post-physics setup.
        self.physics_manager.register_callback(
            lambda _payload: self._render_context.ensure_initialize(),
            PhysicsEvent.PHYSICS_READY,
            order=5,
        )

        type(self)._instance = self  # Mark as valid singleton only after successful init

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
        return bool(self.get_setting("/isaaclab/visualizer/types")) or bool(
            self.get_setting("/isaaclab/video/auto_start_kit")
        )

    def is_headless_or_exist_active_visualizer(self) -> bool:
        """Return whether the simulation should keep stepping without visualizers or with an active visualizer."""
        return not self._visualizers or any(viz.is_running() and not viz.is_closed for viz in self._visualizers)

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
        """Return whether rgb-array rendering is currently available."""
        return self.has_gui or self.has_offscreen_render or self.has_active_visualizers()

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
            or bool(self.resolve_visualizer_types())
            or self._xr_enabled
        )

    def get_physics_dt(self) -> float:
        """Returns the physics time step."""
        return self.physics_manager.get_physics_dt()

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

    def _create_default_visualizer_configs(self, requested_visualizers: list[str]) -> list:
        """Create default visualizer configs for requested types.

        Loads only the requested visualizer submodule (e.g. isaaclab_visualizers.rerun)
        so dependencies for other backends are not imported.
        """
        import importlib

        default_configs = []
        cfg_class_names = {
            "kit": "KitVisualizerCfg",
            "newton_gl": "NewtonGLVisualizerCfg",
            "newton_rtx": "NewtonRTXVisualizerCfg",
            "rerun": "RerunVisualizerCfg",
            "viser": "ViserVisualizerCfg",
        }
        # newton_gl and newton_rtx both live in the isaaclab_visualizers.newton package.
        module_overrides = {"newton_gl": "isaaclab_visualizers.newton", "newton_rtx": "isaaclab_visualizers.newton"}
        for viz_type in requested_visualizers:
            try:
                # Resolve deprecated aliases before lookup.
                if viz_type in _VISUALIZER_ALIASES:
                    canonical = _VISUALIZER_ALIASES[viz_type]
                    import warnings

                    warnings.warn(
                        f"Visualizer type '{viz_type}' is deprecated. Use '{canonical}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    viz_type = canonical
                if viz_type not in _VISUALIZER_TYPES:
                    logger.warning(
                        f"[SimulationContext] Unknown visualizer type '{viz_type}' requested. "
                        f"Valid types: {', '.join(repr(t) for t in _VISUALIZER_TYPES)}. Skipping."
                    )
                    continue
                mod = importlib.import_module(module_overrides.get(viz_type, f"isaaclab_visualizers.{viz_type}"))
                cfg_cls = getattr(mod, cfg_class_names[viz_type])
                cfg = cfg_cls()
                self._apply_default_visualizer_cfg(cfg)
                default_configs.append(cfg)
            except (ImportError, ModuleNotFoundError) as exc:
                # isaaclab_visualizers is optional; log once at warning level
                if "isaaclab_visualizers" in str(exc):
                    logger.warning(
                        "[SimulationContext] Visualizer '%s' skipped: isaaclab_visualizers is not installed. %s",
                        viz_type,
                        _get_visualizer_install_hint(viz_type),
                    )
                else:
                    logger.error(
                        "[SimulationContext] Failed to create default config for visualizer '%s': %s",
                        viz_type,
                        exc,
                    )
            except Exception as exc:
                logger.error(f"[SimulationContext] Failed to create default config for visualizer '{viz_type}': {exc}")
        return default_configs

    def _apply_default_visualizer_cfg(self, cfg: Any) -> None:
        """Apply shared default visualizer settings to a backend-specific config.

        Only propagates fields that were **explicitly set** in ``default_visualizer_cfg``
        (i.e. differ from the base :class:`~isaaclab.visualizers.VisualizerCfg` defaults)
        AND are still at the backend cfg's own class default (i.e. not already
        customised by the caller).  This prevents base-class defaults such as
        ``streaming_view=False`` from stomping backend-specific defaults like
        ``NewtonGLVisualizerCfg.streaming_view=True``.
        """
        from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

        default_cfg = getattr(self.cfg, "default_visualizer_cfg", None)
        if default_cfg is None:
            return
        # Base VisualizerCfg defaults — used to detect which fields on default_cfg
        # were explicitly set by the env vs. left at the base-class default.
        try:
            base_defaults = VisualizerCfg()
        except Exception:
            base_defaults = None
        # Backend-specific class defaults — used to detect which fields on cfg
        # the caller has already customised beyond the class defaults.
        try:
            factory_defaults = type(cfg)()
        except Exception:
            factory_defaults = None
        for field in fields(default_cfg):
            if field.name in ("class_type", "visualizer_type") or not hasattr(cfg, field.name):
                continue
            default_val = getattr(default_cfg, field.name)
            # Skip fields that were not explicitly set in default_cfg (still at base default).
            if base_defaults is not None and hasattr(base_defaults, field.name):
                if default_val == getattr(base_defaults, field.name):
                    continue
            # Preserve explicitly customised fields on cfg.  When factory_defaults is None
            # (backend cfg constructor raised), skip the field rather than overwriting it
            # unconditionally — we cannot tell whether the caller customised it.
            if factory_defaults is None:
                continue
            if getattr(cfg, field.name) != getattr(factory_defaults, field.name):
                continue
            setattr(cfg, field.name, default_val)

    def _get_cli_visualizer_types(self) -> list[str]:
        """Return list of visualizer types requested via CLI (setting)."""
        requested = self.get_setting("/isaaclab/visualizer/types")
        if not isinstance(requested, str) or not requested.strip():
            return []
        # App launcher writes this as a single string; accept comma and/or whitespace separators.
        return [value for chunk in requested.split(",") for value in chunk.split() if value]

    def _apply_visualizer_cli_overrides(self, visualizer_cfgs: list[Any]) -> None:
        """Apply ``--max_visible_envs`` to every resolved visualizer cfg when set in settings.

        AppLauncher stores ``/isaaclab/visualizer/max_visible_envs`` as ``-1`` when the flag was
        omitted; any non-negative int overrides :attr:`VisualizerCfg.max_visible_envs` on each cfg.
        """
        raw = self.get_setting("/isaaclab/visualizer/max_visible_envs")
        try:
            max_visible = int(raw) if raw is not None else -1
        except (TypeError, ValueError):
            logger.warning("[SimulationContext] Invalid /isaaclab/visualizer/max_visible_envs: %r", raw)
            return
        if max_visible < 0:
            return
        for cfg in visualizer_cfgs:
            if hasattr(cfg, "max_visible_envs"):
                cfg.max_visible_envs = max_visible

    def _is_cli_visualizer_explicit(self) -> bool:
        """Return ``True`` when visualizers were explicitly provided via CLI."""
        return bool(self.get_setting("/isaaclab/visualizer/explicit"))

    def _is_cli_visualizer_disable_all(self) -> bool:
        """Return ``True`` when CLI requested ``--viz none`` semantics."""
        return bool(self.get_setting("/isaaclab/visualizer/disable_all"))

    def resolve_visualizer_types(self) -> list[str]:
        """Resolve visualizer types from config or CLI settings."""
        if self._is_cli_visualizer_disable_all():
            return []
        if self._is_cli_visualizer_explicit():
            return self._get_cli_visualizer_types()

        visualizer_cfgs = self.cfg.visualizer_cfgs
        if visualizer_cfgs is None:
            return []
        if not isinstance(visualizer_cfgs, list):
            visualizer_cfgs = [visualizer_cfgs]
        return [cfg.visualizer_type for cfg in visualizer_cfgs if getattr(cfg, "visualizer_type", None)]

    def _resolve_visualizer_cfgs(self) -> list[Any]:
        """Resolve final visualizer configs from cfg and optional CLI override.

        When visualizers are explicitly requested via ``--visualizer`` CLI flag,
        a :class:`RuntimeError` is raised if any requested type cannot be
        resolved (unknown type or missing package).
        """
        visualizer_cfgs: list[Any] = []
        if self.cfg.visualizer_cfgs is not None:
            visualizer_cfgs = (
                self.cfg.visualizer_cfgs if isinstance(self.cfg.visualizer_cfgs, list) else [self.cfg.visualizer_cfgs]
            )

        cli_requested = self._get_cli_visualizer_types()
        cli_explicit = self._is_cli_visualizer_explicit()
        cli_disable_all = self._is_cli_visualizer_disable_all()

        if cli_disable_all:
            resolved = []
        elif not cli_explicit:
            for cfg in visualizer_cfgs:
                self._apply_default_visualizer_cfg(cfg)
            self._apply_visualizer_cli_overrides(visualizer_cfgs)
            resolved = visualizer_cfgs
        elif not visualizer_cfgs:
            resolved = self._create_default_visualizer_configs(cli_requested) if cli_requested else []
            self._apply_visualizer_cli_overrides(resolved)
        else:
            # CLI selection is explicit: keep only requested cfg types, then add defaults for missing.
            cli_requested_set = set(cli_requested)
            resolved = [cfg for cfg in visualizer_cfgs if getattr(cfg, "visualizer_type", None) in cli_requested_set]
            for cfg in resolved:
                self._apply_default_visualizer_cfg(cfg)
            existing_types = {getattr(cfg, "visualizer_type", None) for cfg in resolved}
            for viz_type in cli_requested:
                if viz_type not in existing_types and viz_type in _VISUALIZER_TYPES:
                    resolved.extend(self._create_default_visualizer_configs([viz_type]))
                    existing_types.add(viz_type)
            self._apply_visualizer_cli_overrides(resolved)

        # When visualizers were explicitly requested via CLI, verify all
        # requested types were resolved.  This catches unknown types and
        # missing packages that _create_default_visualizer_configs silently
        # skips.
        if cli_explicit and cli_requested:
            resolved_types = {getattr(cfg, "visualizer_type", None) for cfg in resolved}
            missing = [t for t in cli_requested if t not in resolved_types]
            if missing:
                install_hints = " ".join(
                    _get_visualizer_install_hint(visualizer_type)
                    for visualizer_type in missing
                    if visualizer_type in _VISUALIZER_TYPES
                )
                raise RuntimeError(
                    f"Explicitly requested visualizer(s) {missing} could not be configured. "
                    f"Valid types: {', '.join(repr(t) for t in _VISUALIZER_TYPES)}. "
                    f"{install_hints}"
                )

        # XR auto-start: auto-inject a KitVisualizer when XR is active and no
        # Kit visualizer is already present.  The KitVisualizer pumps
        # app.update() and triggers forward() (via requires_forward_before_step)
        # to sync Fabric data so the XR runtime receives up-to-date hand/joint
        # transforms each frame.
        if self._xr_enabled and bool(self.get_setting("/isaaclab/xr/auto_start")):
            has_kit = any(getattr(cfg, "visualizer_type", None) == "kit" for cfg in resolved)
            if not has_kit:
                try:
                    import importlib

                    mod = importlib.import_module("isaaclab_visualizers.kit")
                    kit_cfg_cls = getattr(mod, "KitVisualizerCfg")
                    resolved.append(kit_cfg_cls())
                    logger.info("[SimulationContext] Auto-injecting KitVisualizer for XR app-update pumping.")
                except (ImportError, ModuleNotFoundError, AttributeError) as exc:
                    logger.warning(
                        "[SimulationContext] XR mode could not auto-inject a KitVisualizer: %s. %s",
                        exc,
                        _get_visualizer_install_hint("kit"),
                    )

        return resolved

    def initialize_visualizers(self) -> None:
        """Initialize visualizers from ``SimulationCfg.visualizer_cfgs``."""
        if self._pending_visualizer_cfgs == [] or (self._pending_visualizer_cfgs is None and self._visualizers):
            return

        visualizer_cfgs = self._get_visualizer_cfgs()
        if not visualizer_cfgs:
            return

        self._initialize_visualizers()

        if not self._visualizers and self._scene_data_provider is not None:
            close_provider = getattr(self._scene_data_provider, "close", None)
            if callable(close_provider):
                close_provider()
            self._scene_data_provider = None

    def _get_visualizer_cfgs(self) -> list[Any]:
        """Resolve visualizer configs for the current initialization cycle."""
        if self._pending_visualizer_cfgs is None:
            self._pending_visualizer_cfgs = self._resolve_visualizer_cfgs()
        return self._pending_visualizer_cfgs

    def _initialize_visualizers(self, config_filter: Callable[[Any], bool] | None = None) -> None:
        """Initialize pending visualizers, optionally restricted by config."""
        physics_dt = getattr(self.cfg.physics, "dt", None)
        self._viz_dt = (physics_dt if physics_dt is not None else self.cfg.dt) * self.cfg.render_interval

        visualizer_cfgs = self._get_visualizer_cfgs()
        if not visualizer_cfgs:
            return

        cli_explicit = self._is_cli_visualizer_explicit()

        configs = [viz.cfg for viz in self._visualizers] + visualizer_cfgs
        for config in configs:
            if config.visualizer_type is not None:
                requires_stage, requires_model = REQUIRES_STAGE_AND_MODEL[config.visualizer_type]
                self.requires_usd_stage |= requires_stage
                self.requires_newton_model |= requires_model

        pending_cfgs = []
        new_visualizers = []
        for cfg in visualizer_cfgs:
            if config_filter is not None and not config_filter(cfg):
                pending_cfgs.append(cfg)
                continue
            try:
                visualizer = cfg.class_type(cfg)
                visualizer.initialize(self._scene_data_provider)
                self._visualizers.append(visualizer)
                new_visualizers.append(visualizer)
            except Exception as exc:
                if cli_explicit:
                    raise RuntimeError(
                        f"Visualizer '{cfg.visualizer_type}' was explicitly requested "
                        f"but failed to create or initialize: {exc}"
                    ) from exc
                logger.exception(
                    "Failed to initialize visualizer '%s' (%s): %s",
                    cfg.visualizer_type,
                    type(cfg).__name__,
                    exc,
                )
        self._pending_visualizer_cfgs = pending_cfgs

        # Replay any camera pose requested before visualizers were initialized.
        pending = getattr(self, "_pending_camera_view", None)
        if pending is not None:
            eye, target = pending
            for viz in new_visualizers:
                viz.set_camera_view(eye, target)
            if not pending_cfgs:
                self._pending_camera_view = None

    def get_scene_data_provider(self) -> SceneDataProvider:
        return self._scene_data_provider

    def register_interactive_scene(self, scene) -> None:
        """Register the active scene so scene data providers can expose scene-owned sensors."""
        self._interactive_scene = scene
        if self._scene_data_provider is not None:
            self._scene_data_provider.set_interactive_scene(scene)

    def get_clone_plan(self) -> ClonePlan | None:
        """Return the clone plan published by the scene.

        Set after replication. Consumed by scene data providers that build backend models
        (e.g. Newton visualizer model on a PhysX backend) from the same plan the cloner used.
        ``None`` until the scene replicates.
        """
        return self._clone_plan

    def set_clone_plan(self, plan: ClonePlan | None) -> None:
        """Set the cloner's clone plan."""
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
        self._initialize_visualizers(self._requires_pre_capture_newton_init)
        for viz in (viz for viz in self._visualizers if self._requires_pre_capture_newton_init(viz.cfg)):
            viz.reset(soft=False)

    @staticmethod
    def _requires_pre_capture_newton_init(cfg: Any) -> bool:
        """Return whether a config contributes Newton picking inputs to capture."""
        return (
            getattr(cfg, "visualizer_type", None) in {"newton_gl", "newton_rtx"}
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
            viz.supports_markers() or (viz.supports_live_plots() and getattr(viz.cfg, "enable_live_plots", True))
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
                    if viz.is_closed:
                        logger.info("Visualizer closed: %s", type(viz).__name__)
                    else:
                        logger.info("Visualizer not running: %s", type(viz).__name__)
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
        if visualizers_to_remove and not self._visualizers:
            self._pending_visualizer_cfgs = None

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
        self._settings_helper.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Get a setting value."""
        return self._settings_helper.get(name)

    def get_or_create_backend(
        self,
        backend_type: type[_BackendT],
        *args: Any,
        clone_role: Literal["physics", "model", "scene"] | None = None,
        **kwargs: Any,
    ) -> _BackendT:
        """Return the simulation-scoped native backend for a type.

        Consumers that register the same backend type resolve one shared native resource
        instead of constructing state to synchronize.

        Args:
            backend_type: Backend class to construct when the resource does not exist.
            *args: Positional arguments used only when constructing the resource.
            clone_role: ``"physics"`` for native physics replication, ``"model"`` for model
                construction, or ``"scene"`` for whole-scene materialization.
            **kwargs: Keyword arguments used only when constructing the resource.

        Returns:
            The existing or newly constructed native backend.
        """
        if clone_role not in (None, "physics", "model", "scene"):
            raise ValueError(f"Unknown clone role: {clone_role!r}.")
        if backend_type not in self._backend_registry:
            self._backend_registry[backend_type] = backend_type(*args, **kwargs)
        if clone_role is not None:
            self._backend_clone_roles.setdefault(backend_type, set()).add(clone_role)
        return cast(_BackendT, self._backend_registry[backend_type])

    @classmethod
    def clear_instance(cls) -> None:
        """Clean up resources and clear the singleton instance."""
        instance = cls._instance
        if instance is not None:
            teardown_errors: list[Exception] = []

            def run_cleanup(callback: Callable[[], Any]) -> None:
                try:
                    callback()
                except Exception as exc:
                    teardown_errors.append(exc)

            try:
                # Close physics manager FIRST to detach PhysX from the stage.
                run_cleanup(instance.physics_manager.close)

                # Close camera renderers after STOP invalidates camera-owned render data and
                # before the stage is closed so stage-bound renderer resources remain valid.
                run_cleanup(instance._render_context.close)

                # Give every visualizer a chance to release its resources.
                for viz in list(instance._visualizers):
                    run_cleanup(viz.close)
                instance._visualizers.clear()

                for resource in instance._backend_registry.values():
                    if (clear := getattr(resource, "clear", None)) is not None:
                        run_cleanup(clear)
                instance._backend_registry.clear()
                instance._backend_clone_roles.clear()

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
            path = prim.GetPath().pathString
            if path == "/World":
                return False
            if prim.GetTypeName() == "PhysicsScene":
                return False
            return True

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
    visualizers: list[str] | None = None,
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
        visualizers: List of visualizer backend keys to enable (e.g. ``["kit", "newton_gl", "rerun"]``).
            Valid types: ``"kit"``, ``"newton_gl"``, ``"newton_rtx"``, ``"rerun"``, ``"viser"``.
            ``"newton"`` is a deprecated alias for ``"newton_gl"``.
            When provided, sets the ``/isaaclab/visualizer/types`` setting so the
            existing visualizer resolution machinery picks them up. Defaults to None.

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

        if visualizers:
            sim.set_setting("/isaaclab/visualizer/types", " ".join(visualizers))

        if add_ground_plane:
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and (sim.get_setting("/isaaclab/has_gui") or visualizers)):
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
            if not sim.get_setting("/isaaclab/has_gui"):
                sim.stop()
            sim.clear_instance()
