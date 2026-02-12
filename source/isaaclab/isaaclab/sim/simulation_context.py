# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

import carb
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.stage as stage_utils
from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils import create_new_stage_in_memory
from isaaclab.visualizers import Visualizer

from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg

logger = logging.getLogger(__name__)


class SettingsHelper:
    """Helper for typed Carbonite settings access."""

    def __init__(self, settings: carb.settings.ISettings):
        self._settings = settings

    def set(self, name: str, value: Any) -> None:
        """Set a Carbonite setting with automatic type routing."""
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
        """Get a Carbonite setting value."""
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

        # Store config
        self.cfg = SimulationCfg() if cfg is None else cfg

        # Get or create stage based on config
        stage_cache = UsdUtils.StageCache.Get()
        if self.cfg.create_stage_in_memory:
            # Create a fresh in-memory stage (not attached to USD context)
            self.stage = create_new_stage_in_memory()
        else:
            # Use existing stage from cache, or create in-memory as fallback
            all_stages = stage_cache.GetAllStages() if stage_cache.Size() > 0 else []  # type: ignore[union-attr]
            self.stage = all_stages[0] if all_stages else create_new_stage_in_memory()

        # Cache stage in USD cache
        stage_id = stage_cache.GetId(self.stage).ToLongInt()  # type: ignore[union-attr]
        if stage_id < 0:
            stage_cache.Insert(self.stage)  # type: ignore[union-attr]

        # Set as current stage in thread-local context for get_current_stage()
        stage_utils._context.stage = self.stage

        # Acquire settings interface and create helper
        self._carb_settings = carb.settings.get_settings()
        self._settings_helper = SettingsHelper(self._carb_settings)

        # Initialize USD physics scene and physics manager
        self._init_usd_physics_scene()
        # Set default physics backend if not specified
        if self.cfg.physics is None:
            from isaaclab_physx.physics import PhysxCfg

            self.cfg.physics = PhysxCfg()
        self._physics = self.cfg.physics
        self.physics_manager: type[PhysicsManager] = self._physics.class_type
        self.physics_manager.initialize(self)

        # Initialize visualizers
        self._init_visualizers()

        # Cache commonly-used settings (these don't change during runtime)
        self._has_gui = bool(self.get_setting("/isaaclab/has_gui"))
        self._has_offscreen_render = bool(self.get_setting("/isaaclab/render/offscreen"))
        # Note: has_rtx_sensors is NOT cached because it changes when Camera sensors are created

        # Simulation state
        self._is_playing = False
        self._is_stopped = True
        type(self)._instance = self  # Mark as valid singleton only after successful init

    def _init_usd_physics_scene(self) -> None:
        """Create and configure the USD physics scene."""
        cfg = self.cfg
        with sim_utils.use_stage(self.stage):
            # Set stage conventions for metric units
            UsdGeom.SetStageUpAxis(self.stage, "Z")
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)

            # Find and delete any existing physics scene
            for prim in self.stage.Traverse():
                if prim.GetTypeName() == "PhysicsScene":
                    sim_utils.delete_prim(prim.GetPath().pathString, stage=self.stage)

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

    @property
    def is_rendering(self) -> bool:
        """Returns whether rendering is active (GUI or RTX sensors)."""
        return self._has_gui or self.get_setting("/isaaclab/render/rtx_sensors")

    def get_physics_dt(self) -> float:
        """Returns the physics time step."""
        return self.physics_manager.get_physics_dt()

    # VISUALIZER MANAGEMENT
    def _init_visualizers(self) -> None:
        """Initialize visualizers based on config and settings."""
        self._visualizers: list[Visualizer] = []
        self._viz_dt = self.cfg.dt * self.cfg.render_interval

        # Determine which visualizers to create
        viz_str = "omniverse"  # Default
        requested = [v.strip() for v in viz_str.split(",") if v.strip()]

        if len(requested) > 0:
            # Get or create visualizer configs
            cfg_list = self.cfg.visualizer_cfgs
            from isaaclab_physx.visualizers import OVVisualizerCfg

            type_map = {"omniverse": OVVisualizerCfg}
            viz_cfgs = []
            if cfg_list is None:
                for viz_type in requested:
                    viz_cfgs.append(type_map[viz_type]())
            else:
                viz_cfgs = cfg_list if isinstance(cfg_list, list) else [cfg_list]

            # Create and initialize each visualizer
            for cfg in viz_cfgs:
                self._visualizers.append(cfg.create_visualizer())
                # build scene data for visualizer initialization
                if cfg.visualizer_type in ("newton", "rerun"):
                    scene_data = {"scene_data_provider": None}
                elif cfg.visualizer_type == "omniverse":
                    scene_data = {"usd_stage": self.stage, "simulation_context": self}
                else:
                    scene_data = {}
                self._visualizers[-1].initialize(scene_data)
                logger.info(f"Initialized visualizer: {type(self._visualizers[-1]).__name__}")

    @property
    def visualizers(self) -> list[Visualizer]:
        """Returns the list of active visualizers."""
        return self._visualizers

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step."""
        return self._viz_dt

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view on all visualizers that support it."""
        for viz in self._visualizers:
            viz.set_camera_view(eye, target)

    def forward(self) -> None:
        """Update kinematics without stepping physics."""
        self.physics_manager.forward()

    def reset(self, soft: bool = False) -> None:
        """Reset the simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        self.physics_manager.reset(soft)
        for viz in self._visualizers:
            viz.reset(soft)
        # Start the timeline so the play button is pressed
        self.physics_manager.play()
        self._is_playing = True
        self._is_stopped = False

    def step(self, render: bool = True) -> None:
        """Step physics, update visualizers, and optionally render.

        Args:
            render: Whether to render the scene after stepping. Defaults to True.
        """
        self.physics_manager.step()
        if render:
            self.render()

    def render(self, mode: int | None = None) -> None:
        """Render the scene via all active visualizers."""
        self.physics_manager.forward()
        for viz in self._visualizers:
            if not viz.is_rendering_paused() and viz.is_running():
                viz.step(self.get_rendering_dt(), state=None)
        # Call render callbacks
        if hasattr(self, "_render_callbacks"):
            for callback in self._render_callbacks.values():
                callback(None)  # Pass None as event data

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

    def is_playing(self) -> bool:
        """Returns True if simulation is playing (not paused or stopped)."""
        return self._is_playing

    def is_stopped(self) -> bool:
        """Returns True if simulation is stopped (not just paused)."""
        return self._is_stopped

    def set_setting(self, name: str, value: Any) -> None:
        """Set a Carbonite setting value."""
        self._settings_helper.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Get a Carbonite setting value."""
        return self._settings_helper.get(name)

    @classmethod
    def clear_instance(cls) -> None:
        """Clean up resources and clear the singleton instance."""
        if cls._instance is not None:
            # Close physics manager FIRST to detach PhysX from the stage
            # This must happen before clearing USD prims to avoid PhysX cleanup errors
            cls._instance.physics_manager.close()

            # Now safe to clear stage contents (PhysX is detached)
            cls.clear_stage()

            # Close all visualizers
            for viz in cls._instance._visualizers:
                viz.close()
            cls._instance._visualizers.clear()

            # Remove stage from cache
            stage_cache = UsdUtils.StageCache.Get()
            stage_id = stage_cache.GetId(cls._instance.stage).ToLongInt()  # type: ignore[union-attr]
            if stage_id > 0:
                stage_cache.Erase(cls._instance.stage)  # type: ignore[union-attr]

            # Clear thread-local stage context
            if hasattr(stage_utils._context, "stage"):
                delattr(stage_utils._context, "stage")

            # Clear instance
            cls._instance = None

            gc.collect()
            logger.info("SimulationContext cleared")

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
    device: str = "cuda:0",
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
        device: Device to run the simulation on. Defaults to "cuda:0".
        dt: Time step for the simulation. Defaults to 0.01.
        sim_cfg: SimulationCfg to use. Defaults to None.
        add_ground_plane: Whether to add a ground plane. Defaults to False.
        add_lighting: Whether to add a dome light. Defaults to False.
        auto_add_lighting: Whether to auto-add lighting if GUI present. Defaults to False.

    Yields:
        The simulation context to use for the simulation.
    """
    try:
        if create_new_stage:
            stage_utils.create_new_stage()

        if sim_cfg is None:
            gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
            sim_cfg = SimulationCfg(device=device, dt=dt, gravity=gravity)

        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim.get_setting("/isaaclab/has_gui")):
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=10000
            )
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        if not sim.get_setting("/isaaclab/has_gui"):
            sim.stop()
        sim.clear_instance()
