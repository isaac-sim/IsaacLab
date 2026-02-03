# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import gc
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import carb
from pxr import Usd, UsdUtils

import isaaclab.sim.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import create_new_stage_in_memory, raise_callback_exception_if_any
from isaaclab.utils.version import get_isaac_sim_version
from .interface import PhysicsInterface, VisualizerInterface, Interface
from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg

# import logger
logger = logging.getLogger(__name__)


class SettingsHelper:
    """Helper for typed Carbonite settings access."""

    def __init__(self, settings: "carb.settings.ISettings"):
        self._settings = settings

    def set(self, name: str, value: Any) -> None:
        """Set a Carbonite setting with automatic type routing.

        Args:
            name: The setting name (e.g., "/physics/cudaDevice").
            value: The value to set (bool, int, float, str, list, or tuple).
        """
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

    # Singleton instance
    _instance: "SimulationContext | None" = None

    def __new__(cls, cfg: SimulationCfg | None = None):
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def instance(cls) -> "SimulationContext | None":
        """Get the singleton instance, or None if not created."""
        return cls._instance

    def __init__(self, cfg: SimulationCfg | None = None):
        """Initialize the simulation context.

        Args:
            cfg: Simulation configuration. Defaults to None (uses default config).
        """
        # Skip initialization if already initialized
        if not self._initialized:
            # store input
            self.cfg = SimulationCfg() if cfg is None else cfg
            self.device = self.cfg.device

            # get existing stage or create new one in memory
            stage_cache = UsdUtils.StageCache.Get()
            all_stages = stage_cache.GetAllStages() if stage_cache.Size() > 0 else []
            self.stage = all_stages[0] if all_stages else create_new_stage_in_memory()

            # Cache stage in USD cache
            stage_id = stage_cache.GetId(self.stage).ToLongInt()  # type: ignore[union-attr]
            if stage_id < 0:
                stage_cache.Insert(self.stage)  # type: ignore[union-attr]

            # Acquire settings interface and create helper
            self.carb_settings = carb.settings.get_settings()
            self._settings_helper = SettingsHelper(self.carb_settings)

            # Initialize interfaces (order matters: visualizer first for config, then physics)
            self._visualizer_interface = VisualizerInterface(self)
            self._physics_interface = PhysicsInterface(self)

            # List of interfaces for common operations
            self._interfaces: list[Interface] = [
                self._physics_interface,
                self._visualizer_interface,
            ]

            # define a global variable to store the exceptions raised in the callback stack
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None

            self._is_playing = False
            self._initialized = True

    def _call_interfaces(self, method: str, **kwargs) -> None:
        """Call a method on all interfaces."""
        for interface in self._interfaces:
            getattr(interface, method)(**kwargs)
        raise_callback_exception_if_any()

    def get_version(self) -> tuple[int, int, int]:
        """Returns the version of the simulator (major, minor, patch)."""
        ver = get_isaac_sim_version()
        return ver.major, ver.minor, ver.micro

    def set_setting(self, name: str, value: Any) -> None:
        """Set a Carbonite setting value."""
        self._settings_helper.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Get a Carbonite setting value."""
        return self._settings_helper.get(name)

    def forward(self) -> None:
        """Update kinematics and sync scene data without stepping physics."""
        self._call_interfaces("forward")

    def reset(self, soft: bool = False) -> None:
        """Reset the simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        self._call_interfaces("reset", soft=soft)
        self._is_playing = True

    def step(self, render: bool = True) -> None:
        """Step physics, update visualizers, and optionally render.

        Args:
            render: Whether to render the scene after stepping. Defaults to True.
        """
        self._call_interfaces("step", render=render)

    def is_playing(self) -> bool:
        """Returns True if simulation is playing."""
        return self._is_playing

    def is_stopped(self) -> bool:
        """Returns True if simulation is stopped."""
        return not self._is_playing

    def play(self) -> None:
        """Start the simulation."""
        self._call_interfaces("play")
        self._is_playing = True

    def pause(self) -> None:
        """Pause the simulation."""
        self._call_interfaces("pause")

    def stop(self) -> None:
        """Stop the simulation."""
        self._call_interfaces("stop")
        self._is_playing = False

    def render(self, mode: int | None = None) -> None:
        """Refresh rendering components (viewports, UI).

        Args:
            mode: Render mode. Defaults to None (use current mode).
        """
        self._physics_interface.forward()
        self._visualizer_interface.render()

    def get_physics_dt(self) -> float:
        """Returns the physics time step."""
        return self._physics_interface.physics_dt

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step."""
        return self._visualizer_interface.get_rendering_dt()

    @classmethod
    def clear_instance(cls, clear_stage: bool = True) -> None:
        """Clean up resources and clear the singleton instance.

        Args:
            clear_stage: If True, clear stage prims (preserving /World and PhysicsScene)
                before closing. Defaults to True.
        """
        if cls._instance is None:
            return

        if not cls._instance._initialized:
            logger.warning("Simulation context not initialized.")
            return

        # Optionally clear stage contents first
        if clear_stage:
            cls.clear_stage()
        # Close all interfaces
        cls._instance._call_interfaces("close")

        # Remove stage from cache
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(cls._instance.stage).ToLongInt()  # type: ignore[union-attr]
        if stage_id > 0:
            stage_cache.Erase(cls._instance.stage)  # type: ignore[union-attr]

        # Clear instance
        cls._instance._initialized = False
        cls._instance = None

        gc.collect()

    @classmethod
    def clear_stage(cls) -> None:
        """Clear the current USD stage (preserving /World and PhysicsScene).

        Note:
            This clears stage prims but keeps the context alive. For full cleanup,
            use :meth:`clear_instance` instead.
        """
        if cls._instance is None:
            return

        def _predicate(prim: Usd.Prim) -> bool:
            path = prim.GetPath().pathString  # type: ignore[union-attr]
            return path != "/World" and prim.GetTypeName() != "PhysicsScene"

        sim_utils.clear_stage(stage=cls._instance.stage, predicate=_predicate)


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

    This function facilitates the creation of a simulation context and provides flexibility in configuring various
    aspects of the simulation, such as time step, gravity, device, and scene elements like ground plane and
    lighting.

    If :attr:`sim_cfg` is None, then an instance of :class:`SimulationCfg` is created with default settings, with parameters
    overwritten based on arguments to the function.

    An example usage of the context manager function:

    ..  code-block:: python

        with build_simulation_context() as sim:
             # Design the scene

             # Play the simulation
             sim.reset()
             while sim.is_playing():
                 sim.step()

    Args:
        create_new_stage: Whether to create a new stage. Defaults to True.
        gravity_enabled: Whether to enable gravity in the simulation. Defaults to True.
        device: Device to run the simulation on. Defaults to "cuda:0".
        dt: Time step for the simulation: Defaults to 0.01.
        sim_cfg: :class:`isaaclab.sim.SimulationCfg` to use for the simulation. Defaults to None.
        add_ground_plane: Whether to add a ground plane to the simulation. Defaults to False.
        add_lighting: Whether to add a dome light to the simulation. Defaults to False.
        auto_add_lighting: Whether to automatically add a dome light to the simulation if the simulation has a GUI.
            Defaults to False. This is useful for debugging tests in the GUI.

    Yields:
        The simulation context to use for the simulation.

    """
    try:
        if create_new_stage:
            stage_utils.create_new_stage()

        if sim_cfg is None:
            from isaaclab.physics.physx_manager_cfg import PhysxManagerCfg

            gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
            physics_manager_cfg = PhysxManagerCfg(dt=dt, device=device, gravity=gravity)
            sim_cfg = SimulationCfg(physics_manager_cfg=physics_manager_cfg)

        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            # Ground-plane
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim.get_setting("/isaaclab/has_gui")):
            # Lighting
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=10000
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        if not sim.get_setting("/isaaclab/has_gui"):
            sim.stop()
        sim.clear_instance()
