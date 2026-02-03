# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdUtils

import isaaclab.sim as sim_utils
from isaaclab.utils.logger import configure_logging
from isaaclab.utils.version import get_isaac_sim_version

from .physics_interface import PhysicsInterface
from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg
from .visualizer_interface import VisualizerInterface

logger = logging.getLogger(__name__)


class SimulationContext:
    """A class to control simulation-related events such as physics stepping and rendering.

    The simulation context helps control various simulation aspects. This includes:

    * configure the simulator with different settings such as the physics time-step, the number of physics substeps,
      and the physics solver parameters (for more information, see :class:`isaaclab.sim.SimulationCfg`)
    * playing, pausing, stepping and stopping the simulation
    * adding and removing callbacks to different simulation events such as physics stepping, rendering, etc.

    This class provides a standalone simulation context for Isaac Lab, with additional functionalities
    such as setting up the simulation context with a configuration object, exposing commonly used
    simulator-related functions, and performing version checks of Isaac Sim to ensure compatibility
    between releases.

    The simulation context is a singleton object. This means that there can only be one instance
    of the simulation context at any given time. This is enforced by the parent class. Therefore, it is
    not possible to create multiple instances of the simulation context. Instead, the simulation context
    can be accessed using the ``instance()`` method.

    .. attention::
        Since we only support the `PyTorch <https://pytorch.org/>`_ backend for simulation, the
        simulation context is configured to use the ``torch`` backend by default. This means that
        all the data structures used in the simulation are ``torch.Tensor`` objects.

    The simulation context can be used in two different modes of operations:

    1. **Standalone python script**: In this mode, the user has full control over the simulation and
       can trigger stepping events synchronously (i.e. as a blocking call). In this case the user
       has to manually call :meth:`step` step the physics simulation and :meth:`render` to
       render the scene.
    2. **Omniverse extension**: In this mode, the user has limited control over the simulation stepping
       and all the simulation events are triggered asynchronously (i.e. as a non-blocking call). In this
       case, the user can only trigger the simulation to start, pause, and stop. The simulation takes
       care of stepping the physics simulation and rendering the scene.

    Based on above, for most functions in this class there is an equivalent function that is suffixed
    with ``_async``. The ``_async`` functions are used in the Omniverse extension mode and
    the non-``_async`` functions are used in the standalone python script mode.
    """

    _instance: SimulationContext | None = None
    """The singleton instance of the simulation context."""

    _is_initialized: ClassVar[bool] = False
    """Whether the simulation context is initialized."""

    def __init__(self, cfg: SimulationCfg | None = None):
        """Creates a simulation context to control the simulator.

        Args:
            cfg: The configuration of the simulation. Defaults to None,
                in which case the default configuration is used.
        """
        # skip if already initialized
        if SimulationContext._is_initialized:
            return
        SimulationContext._is_initialized = True

        # store input
        if cfg is None:
            cfg = SimulationCfg()
        # check that the config is valid
        cfg.validate()  # type: ignore
        # store configuration
        self.cfg = cfg

        # setup logger
        self.logger = configure_logging(
            logging_level=self.cfg.logging_level,
            save_logs_to_file=self.cfg.save_logs_to_file,
            log_dir=self.cfg.log_dir,
        )

        # create stage in memory if requested
        if self.cfg.create_stage_in_memory:
            self._initial_stage = sim_utils.create_new_stage_in_memory()
        else:
            self._initial_stage = omni.usd.get_context().get_stage()
            if not self._initial_stage:
                self._initial_stage = sim_utils.create_new_stage()
        # check that stage is created
        if self._initial_stage is None:
            raise RuntimeError("Failed to create a new stage. Please check if the USD context is valid.")
        # add stage to USD cache
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(self._initial_stage).ToLongInt()
        if stage_id < 0:
            stage_id = stage_cache.Insert(self._initial_stage).ToLongInt()
        self._initial_stage_id = stage_id

        # acquire settings interface
        self.carb_settings = carb.settings.get_settings()

        # initialize visualizer interface (handles viewport, render mode, render settings)
        self._visualizer = VisualizerInterface(self)
        # define a global variable to store the exceptions raised in the callback stack
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None

        # initialize physics interface (handles scene creation, delegates to backends)
        self._physics_interface = PhysicsInterface(self)

    def __new__(cls, *args, **kwargs) -> SimulationContext:
        """Returns the instance of the simulation context.

        This function is used to create a singleton instance of the simulation context.
        If the instance already exists, it returns the previously defined one. Otherwise,
        it creates a new instance and returns it
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        else:
            logger.debug("Returning the previously defined instance of Simulation Context.")
        return cls._instance  # type: ignore

    """
    Operations - Singleton.
    """

    @classmethod
    def instance(cls) -> SimulationContext | None:
        """Returns the instance of the simulation context.

        Returns:
            The instance of the simulation context. None if the instance is not initialized.
        """
        return cls._instance

    @classmethod
    def clear_instance(cls) -> None:
        """Delete the simulation context singleton instance."""
        if cls._instance is not None:
            # check if the instance is initialized
            if not cls._is_initialized:
                logger.warning("Simulation context is not initialized. Unable to clear the instance.")
                return
            # close visualizer (unsubscribes stop handle)
            cls._instance._visualizer.close()
            # close physics interface (clears PhysxManager, detaches physx stage)
            cls._instance._physics_interface.close()
            # detach the stage from the USD stage cache
            stage_cache = UsdUtils.StageCache.Get()
            stage_id = stage_cache.GetId(cls._instance._initial_stage).ToLongInt()
            if stage_id > 0:
                stage_cache.Erase(cls._instance._initial_stage)
            # clear the instance and the flag
            cls._instance = None
            cls._is_initialized = False

    """
    Properties.
    """

    @property
    def stage(self) -> Usd.Stage:
        """USD Stage."""
        return self._initial_stage

    @property
    def device(self) -> str:
        """Device used by the simulation.

        Note:
            In Omniverse, it is possible to configure multiple GPUs for rendering, while physics engine
            operates on a single GPU. This function returns the device that is used for physics simulation.
        """
        return self._physics_interface.device

    """
    Operations - Simulation Information.
    """
    def get_version(self) -> tuple[int, int, int]:
        """Returns the version of the simulator.

        The returned tuple contains the following information:

        * Major version: This is the year of the release (e.g. 2022).
        * Minor version: This is the half-year of the release (e.g. 1 or 2).
        * Patch version: This is the patch number of the release (e.g. 0).

        .. attention::
            This function is deprecated and will be removed in the future.
            We recommend using :func:`isaaclab.utils.version.get_isaac_sim_version`
            instead of this function.

        Returns:
            A tuple containing the major, minor, and patch versions.

        Example:
            >>> sim = SimulationContext()
            >>> sim.get_version()
            (2022, 1, 0)
        """
        return get_isaac_sim_version().major, get_isaac_sim_version().minor, get_isaac_sim_version().micro

    def get_physics_dt(self) -> float:
        """Returns the physics time step of the simulation.

        Returns:
            The physics time step of the simulation.
        """
        return self.cfg.physics_manager_cfg.dt

    @property
    def physics_prim_path(self) -> str:
        """The path to the physics scene prim."""
        return self.cfg.physics_manager_cfg.physics_prim_path

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step of the simulation.

        Returns:
            The rendering time step of the simulation.
        """
        return self.cfg.physics_manager_cfg.dt * self.cfg.render_interval

    """
    Operations - Utilities.
    """

    def set_camera_view(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        """Set the location and target of the viewport camera in the stage.

        Note:
            This is a wrapper around the :math:`isaacsim.core.utils.viewports.set_camera_view` function.
            It is provided here for convenience to reduce the amount of imports needed.

        Args:
            eye: The location of the camera eye.
            target: The location of the camera target.
            camera_prim_path: The path to the camera primitive in the stage. Defaults to
                "/OmniverseKit_Persp".
        """
        self._visualizer.set_camera_view(eye, target)

    def set_setting(self, name: str, value: Any):
        """Set simulation settings using the Carbonite SDK.

        .. note::
            If the input setting name does not exist, it will be created. If it does exist, the value will be
            overwritten. Please make sure to use the correct setting name.

            To understand the settings interface, please refer to the
            `Carbonite SDK <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/settings.html>`_
            documentation.

        Args:
            name: The name of the setting.
            value: The value of the setting.
        """
        # Route through typed setters for correctness and consistency for common scalar types.
        if isinstance(value, bool):
            self.carb_settings.set_bool(name, value)
        elif isinstance(value, int):
            self.carb_settings.set_int(name, value)
        elif isinstance(value, float):
            self.carb_settings.set_float(name, value)
        elif isinstance(value, str):
            self.carb_settings.set_string(name, value)
        elif isinstance(value, (list, tuple)):
            self.carb_settings.set(name, value)
        else:
            raise ValueError(f"Unsupported value type for setting '{name}': {type(value)}")

    def get_setting(self, name: str) -> Any:
        """Read the simulation setting using the Carbonite SDK.

        Args:
            name: The name of the setting.

        Returns:
            The value of the setting.
        """
        return self.carb_settings.get(name)

    """
    Operations- Timeline.
    """

    def is_playing(self) -> bool:
        """Check whether the simulation is playing.

        Returns:
            True if the simulator is playing.
        """
        return self._visualizer.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped.

        Returns:
            True if the simulator is stopped.
        """
        return self._visualizer.is_stopped()

    def play(self) -> None:
        """Start playing the simulation."""
        self._visualizer.play()
        # check for callback exceptions
        self._check_for_callback_exceptions()

    def pause(self) -> None:
        """Pause the simulation."""
        self._visualizer.pause()
        # check for callback exceptions
        self._check_for_callback_exceptions()

    def stop(self) -> None:
        """Stop the simulation.

        Note:
            Stopping the simulation will lead to the simulation state being lost.
        """
        self._visualizer.stop()
        # check for callback exceptions
        self._check_for_callback_exceptions()

    """
    Operations - Override (standalone)
    """

    def reset(self, soft: bool = False):
        """Reset the simulation.

        .. warning::

            This method is not intended to be used in the Isaac Sim's Extensions workflow since the Kit application
            has the control over the rendering steps. For the Extensions workflow use the ``reset_async`` method instead

        Args:
            soft (bool, optional): if set to True simulation won't be stopped and start again. It only calls the reset on the scene objects.

        """
        self._physics_interface.reset(soft)
        self._visualizer.reset(soft)
        self._check_for_callback_exceptions()

    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        self._physics_interface.forward()

    def step(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                    If set to False, the scene is not rendered and only the physics simulation is stepped.
        """
        self._visualizer.step(render)
        if self.is_playing():
            self._physics_interface.step(render)

    def render(self, mode: RenderMode | None = None):
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
        self._physics_interface.forward()
        self._visualizer.render()

    @classmethod
    def clear(cls):
        """Clear the current USD stage."""

        def _predicate(prim: Usd.Prim) -> bool:
            """Check if the prim should be deleted.

            It adds a check for '/World' and 'PhysicsScene' prims.
            """
            if prim.GetPath().pathString == "/World":
                return False
            if prim.GetTypeName() == "PhysicsScene":
                return False
            return True

        # clear the stage
        if cls._instance is not None:
            stage = cls._instance._initial_stage
            sim_utils.clear_stage(stage=stage, predicate=_predicate)
        else:
            logger.error("Simulation context is not initialized. Unable to clear the stage.")

    """
    Helper Functions
    """

    def _check_for_callback_exceptions(self):
        """Checks for callback exceptions and raises them if found."""
        # disable simulation stopping control so that we can crash the program
        # if an exception is raised in a callback.
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:  # type: ignore
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None  # type: ignore
            raise exception_to_raise


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
            # Clear any existing simulation context before creating a new stage
            # This ensures proper resource cleanup and allows a fresh initialization
            SimulationContext.clear_instance()
            sim_utils.create_new_stage()

        if sim_cfg is None:
            # Construct one and overwrite the dt, gravity, and device in physics_manager_cfg
            from isaaclab.physics.physx_manager_cfg import PhysxManagerCfg

            gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
            physics_manager_cfg = PhysxManagerCfg(dt=dt, device=device, gravity=gravity)
            sim_cfg = SimulationCfg(physics_manager_cfg=physics_manager_cfg)

        # Construct simulation context
        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            # Ground-plane
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim.carb_settings.get("/isaaclab/has_gui")):
            # Lighting
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1),
                enable_color_temperature=True,
                color_temperature=5500,
                intensity=10000,
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        sim.logger.error(traceback.format_exc())
        raise
    finally:
        if not sim.carb_settings.get("/isaaclab/has_gui"):
            # Stop simulation only if we aren't rendering otherwise the app will hang indefinitely
            sim.stop()

        # Clear the stage
        sim.clear_instance()
        # Check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:  # type: ignore
            raise builtins.ISAACLAB_CALLBACK_EXCEPTION  # type: ignore
