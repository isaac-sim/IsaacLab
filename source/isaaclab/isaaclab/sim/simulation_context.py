# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import gc
import logging
import os
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import toml
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.stage as stage_utils
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils import create_new_stage
from isaaclab.visualizers import KitVisualizerCfg, NewtonVisualizerCfg, RerunVisualizerCfg, Visualizer

from .scene_data_providers import SceneDataProvider
from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg

logger = logging.getLogger(__name__)


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


try:
    from isaacsim.core.api.simulation_context import SimulationContext as _SimulationContext
except ImportError:
    class _SimulationContext:
        _instance = None  # so SimulationContext.instance() can access cls._instance when base is this stub


class SimulationContext(_SimulationContext):
    _instance = None  # singleton; ensure attribute exists for instance() and __init__

    """A class to control simulation-related events such as physics stepping and rendering.

    The simulation context helps control various simulation aspects. This includes:

    * configure the simulator with different settings such as the physics time-step, the number of physics substeps,
      and the physics solver parameters (for more information, see :class:`isaaclab.sim.SimulationCfg`)
    * playing, pausing, stepping and stopping the simulation
    * adding and removing callbacks to different simulation events such as physics stepping, rendering, etc.

    This class inherits from the :class:`isaacsim.core.api.simulation_context.SimulationContext` class and
    adds additional functionalities such as setting up the simulation context with a configuration object,
    exposing other commonly used simulator-related functions, and performing version checks of Isaac Sim
    to ensure compatibility between releases.

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

    class RenderMode(enum.IntEnum):
        """Different rendering modes for the simulation.

        Render modes correspond to how the viewport and other UI elements (such as listeners to keyboard or mouse
        events) are updated. There are three main components that can be updated when the simulation is rendered:

        1. **UI elements and other extensions**: These are UI elements (such as buttons, sliders, etc.) and other
           extensions that are running in the background that need to be updated when the simulation is running.
        2. **Cameras**: These are typically based on Hydra textures and are used to render the scene from different
           viewpoints. They can be attached to a viewport or be used independently to render the scene.
        3. **Viewports**: These are windows where you can see the rendered scene.

        Updating each of the above components has a different overhead. For example, updating the viewports is
        computationally expensive compared to updating the UI elements. Therefore, it is useful to be able to
        control what is updated when the simulation is rendered. This is where the render mode comes in. There are
        four different render modes:

        * :attr:`NO_GUI_OR_RENDERING`: The simulation is running without a GUI and off-screen rendering flag
          is disabled, so none of the above are updated.
        * :attr:`NO_RENDERING`: No rendering, where only 1 is updated at a lower rate.
        * :attr:`PARTIAL_RENDERING`: Partial rendering, where only 1 and 2 are updated.
        * :attr:`FULL_RENDERING`: Full rendering, where everything (1, 2, 3) is updated.

        .. _Viewports: https://docs.omniverse.nvidia.com/extensions/latest/ext_viewport.html
        """

        NO_GUI_OR_RENDERING = -1
        """The simulation is running without a GUI and off-screen rendering is disabled."""
        NO_RENDERING = 0
        """No rendering, where only other UI elements are updated at a lower rate."""
        PARTIAL_RENDERING = 1
        """Partial rendering, where the simulation cameras and UI elements are updated."""
        FULL_RENDERING = 2
        """Full rendering, where all the simulation viewports, cameras and UI elements are updated."""

    def __init__(self, cfg: SimulationCfg | None = None):
        """Creates a simulation context to control the simulator.

        Args:
            cfg: The configuration of the simulation. Defaults to None,
                in which case the default configuration is used.
        """
        # store input
        if cfg is None:
            cfg = SimulationCfg()
        # check that the config is valid
        cfg.validate()
        self.cfg = cfg
        # check that simulation is running
        if sim_utils.get_current_stage() is None:
            raise RuntimeError("The stage has not been created. Did you run the simulator?")

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
        # cache stage if it is not already cached
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(self._initial_stage).ToLongInt()
        if stage_id < 0:
            stage_cache.Insert(self._initial_stage)

        # acquire settings interface
        self.carb_settings = carb.settings.get_settings()

        # apply carb physics settings
        self._apply_physics_settings()

        # note: we read this once since it is not expected to change during runtime
        # read flag for whether a local GUI is enabled
        self._local_gui = self.carb_settings.get("/app/window/enabled")
        # read flag for whether livestreaming GUI is enabled
        self._livestream_gui = self.carb_settings.get("/app/livestream/enabled")
        # read flag for whether XR GUI is enabled
        self._xr_gui = self.carb_settings.get("/app/xr/enabled")

        # read flags anim recording config and init timestamps
        self._setup_anim_recording()

        # read flag for whether the Isaac Lab viewport capture pipeline will be used,
        # casting None to False if the flag doesn't exist
        # this flag is set from the AppLauncher class
        self._offscreen_render = bool(self.carb_settings.get("/isaaclab/render/offscreen"))
        # read flag for whether the default viewport should be enabled
        self._render_viewport = bool(self.carb_settings.get("/isaaclab/render/active_viewport"))
        # flag for whether any GUI will be rendered (local, livestreamed or viewport)
        self._has_gui = self._local_gui or self._livestream_gui or self._xr_gui

        # apply render settings from render config
        self._apply_render_settings_from_cfg()

        # store the default render mode
        if not self._has_gui and not self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.NO_GUI_OR_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        elif not self._has_gui and self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.PARTIAL_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        else:
            # note: need to import here in case the UI is not available (ex. headless mode)
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport

            # set default render mode
            # note: this can be changed by calling the `set_render_mode` function
            self.render_mode = self.RenderMode.FULL_RENDERING
            # acquire viewport context
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
            # acquire viewport window
            # TODO @mayank: Why not just use get_active_viewport_and_window() directly?
            self._viewport_window = ui.Workspace.get_window("Viewport")
            # counter for periodic rendering
            self._render_throttle_counter = 0
            # rendering frequency in terms of number of render calls
            self._render_throttle_period = 5

        # check the case where we don't need to render the viewport
        # since render_viewport can only be False in headless mode, we only need to check for offscreen_render
        if not self._render_viewport and self._offscreen_render:
            # disable the viewport if offscreen_render is enabled
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().updates_enabled = False

        # override enable scene querying if rendering is enabled
        # this is needed for some GUI features
        if self._has_gui:
            self.cfg.enable_scene_query_support = True
        # set up flatcache/fabric interface (default is None)
        # this is needed to flush the flatcache data into Hydra manually when calling `render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        # note: need to do this here because super().__init__ calls render and this variable is needed
        self._fabric_iface = None

        # create a tensor for gravity
        # note: this line is needed to create a "tensor" in the device to avoid issues with torch 2.1 onwards.
        #   the issue is with some heap memory corruption when torch tensor is created inside the asset class.
        #   you can reproduce the issue by commenting out this line and running the test `test_articulation.py`.
        self._gravity_tensor = torch.tensor(self.cfg.gravity, dtype=torch.float32, device=self.cfg.device)

        # define a global variable to store the exceptions raised in the callback stack
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None

        # add callback to deal the simulation app when simulation is stopped.
        # this is needed because physics views go invalid once we stop the simulation
        if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
            self._app_control_on_stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP),
                lambda *args, obj=weakref.proxy(self): obj._app_control_on_stop_handle_fn(*args),
                order=15,
            )
        else:
            self._app_control_on_stop_handle = None
        self._disable_app_control_on_stop_handle = False

        # flatten out the simulation dictionary
        sim_params = self.cfg.to_dict()
        if sim_params is not None:
            if "physx" in sim_params:
                physx_params = sim_params.pop("physx")
                sim_params.update(physx_params)

        # add warning about enabling stabilization for large step sizes
        if not self.cfg.physx.enable_stabilization and (self.cfg.dt > 0.0333):
            self.logger.warning(
                "Large simulation step size (> 0.0333 seconds) is not recommended without enabling stabilization."
                " Consider setting the `enable_stabilization` flag to True in the PhysxCfg, or reducing the"
                " simulation step size if you run into physics issues."
            )

        # set simulation device
        # note: Although Isaac Sim sets the physics device in the init function,
        #   it does a render call which gets the wrong device.
        SimulationManager.set_physics_sim_device(self.cfg.device)

        # obtain the parsed device
        # This device should be the same as "self.cfg.device". However, for cases, where users specify the device
        # as "cuda" and not "cuda:X", then it fetches the current device from SimulationManager.
        # Note: Since we fix the device from the configuration and don't expect users to change it at runtime,
        #   we can obtain the device once from the SimulationManager.get_physics_sim_device() function.
        #   This reduces the overhead of calling the function.
        self._physics_device = SimulationManager.get_physics_sim_device()

        # create a simulation context to control the simulator
        if get_isaac_sim_version().major < 5:
            # stage arg is not supported before isaac sim 5.0
            super().__init__(
                stage_units_in_meters=1.0,
                physics_dt=self.cfg.dt,
                rendering_dt=self.cfg.dt * self.cfg.render_interval,
                backend="torch",
                sim_params=sim_params,
                physics_prim_path=self.cfg.physics_prim_path,
                device=self.cfg.device,
            )
        else:
            super().__init__(
                stage_units_in_meters=1.0,
                physics_dt=self.cfg.dt,
                rendering_dt=self.cfg.dt * self.cfg.render_interval,
                backend="torch",
                sim_params=sim_params,
                physics_prim_path=self.cfg.physics_prim_path,
                device=self.cfg.device,
                stage=self._initial_stage,
            )

    """
    Properties - Override.
    """

    @property
    def device(self) -> str:
        """Device used by the simulation.

        Note:
            In Omniverse, it is possible to configure multiple GPUs for rendering, while physics engine
            operates on a single GPU. This function returns the device that is used for physics simulation.
        """
        return self._physics_device

    """
    Operations - New.
    """

    def has_gui(self) -> bool:
        """Returns whether the simulation has a GUI enabled.

        True if the simulation has a GUI enabled either locally or live-streamed.
        """
        return self._has_gui

    def has_rtx_sensors(self) -> bool:
        """Returns whether the simulation has any RTX-rendering related sensors.

        This function returns the value of the simulation parameter ``"/isaaclab/render/rtx_sensors"``.
        The parameter is set to True when instances of RTX-related sensors (cameras or LiDARs) are
        created using Isaac Lab's sensor classes.

        True if the simulation has RTX sensors (such as USD Cameras or LiDARs).

        For more information, please check `NVIDIA RTX documentation`_.

        .. _NVIDIA RTX documentation: https://developer.nvidia.com/rendering-technologies
        """
        return self._settings.get_as_bool("/isaaclab/render/rtx_sensors")

    def is_fabric_enabled(self) -> bool:
        """Returns whether the fabric interface is enabled.

        When fabric interface is enabled, USD read/write operations are disabled. Instead all applications
        read and write the simulation state directly from the fabric interface. This reduces a lot of overhead
        that occurs during USD read/write operations.

        For more information, please check `Fabric documentation`_.

        .. _Fabric documentation: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html
        """
        return self._fabric_iface is not None

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

    """
    Operations - New utilities.
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
        # safe call only if we have a GUI or viewport rendering enabled
        if self._has_gui or self._offscreen_render or self._render_viewport:
            set_camera_view(eye, target, camera_prim_path)

    def set_render_mode(self, mode: RenderMode):
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the different render modes.

        .. note::
            When no GUI is available (locally or livestreamed), we do not need to choose whether the viewport
            needs to render or not (since there is no GUI). Thus, in this case, calling the function will not
            change the render mode.

        Args:
            mode (RenderMode): The rendering mode. If different than SimulationContext's rendering mode,
            SimulationContext's mode is changed to the new mode.

        Raises:
            ValueError: If the input mode is not supported.
        """
        # check if mode change is possible -- not possible when no GUI is available
        if not self._has_gui:
            self.logger.warning(
                f"Cannot change render mode when GUI is disabled. Using the default render mode: {self.render_mode}."
            )
            return
        # check if there is a mode change
        # note: this is mostly needed for GUI when we want to switch between full rendering and no rendering.
        if mode != self.render_mode:
            if mode == self.RenderMode.FULL_RENDERING:
                # display the viewport and enable updates
                self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
                self._viewport_window.visible = True  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == self.RenderMode.PARTIAL_RENDERING:
                # hide the viewport and disable updates
                self._viewport_context.updates_enabled = False  # pyright: ignore [reportOptionalMemberAccess]
                self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == self.RenderMode.NO_RENDERING:
                # hide the viewport and disable updates
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = False  # pyright: ignore [reportOptionalMemberAccess]
                    self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
                # reset the throttle counter
                self._render_throttle_counter = 0
            else:
                raise ValueError(f"Unsupported render mode: {mode}! Please check `RenderMode` for details.")
            # update render mode
            self.render_mode = mode

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

    def reset(self, soft: bool = False) -> None:
        """Reset the simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        self._disable_app_control_on_stop_handle = True
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
        import time as _t
        _t0 = _t.perf_counter()
        super().reset(soft=soft)
        # app.update() may be changing the cuda device in reset, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
        # enable kinematic rendering with fabric
        if self.physics_sim_view:
            _t1 = _t.perf_counter()
            self.physics_sim_view._backend.initialize_kinematic_bodies()
            elapsed = _t.perf_counter() - _t1
            logger.debug("[PERF][simulation_context] reset(): initialize_kinematic_bodies() took %s s", round(elapsed, 3))
        # perform additional rendering steps to warm up replicator buffers
        # this is only needed for the first time we set the simulation
        if not soft:
            for i in range(2):
                _t2 = _t.perf_counter()
                self.render()
                elapsed = _t.perf_counter() - _t2
                logger.debug("[PERF][simulation_context] reset(): render() warmup %s/2 took %s s", i + 1, round(elapsed, 3))
        self._disable_app_control_on_stop_handle = False

    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        if self._fabric_iface is not None:
            if self.physics_sim_view is not None and self.is_playing():
                # Update the articulations' link's poses before rendering
                self.physics_sim_view.update_articulations_kinematic()
            self._update_fabric(0.0, 0.0)

    def step(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                    If set to False, the scene is not rendered and only the physics simulation is stepped.
        """
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise

        # update anim recording if needed
        if self._anim_recording_enabled:
            is_anim_recording_finished = self._update_anim_recording()
            if is_anim_recording_finished:
                logger.warning("[INFO][SimulationContext]: Animation recording finished. Closing app.")
                self._app.shutdown()

        # check if the simulation timeline is paused. in that case keep stepping until it is playing
        if not self.is_playing():
            # step the simulator (but not the physics) to have UI still active
            while not self.is_playing():
                self.render()
                # meantime if someone stops, break out of the loop
                if self.is_stopped():
                    break
            # need to do one step to refresh the app
            # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
            #   without this the app becomes unresponsive.
            # FIXME: This steps physics as well, which we is not good in general.
            self.app.update()

        # step the simulation
        import time as _t

        _t0 = _t.perf_counter()
        super().step(render=render)
        if not hasattr(self, "_step_log_count"):
            self._step_log_count = 0
        self._step_log_count += 1
        if self._step_log_count <= 3 or self._step_log_count % 100 == 0:
            elapsed = _t.perf_counter() - _t0
            logger.debug(
                "[PERF][simulation_context] step(): super().step(render=%s) took %s s (call #%s)",
                render, round(elapsed, 3), self._step_log_count,
            )

        # app.update() may be changing the cuda device in step, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

    def render(self, mode: RenderMode | None = None):
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
        import time as _t

        _t0 = _t.perf_counter()
        # check if we need to change the render mode
        if mode is not None:
            self.set_render_mode(mode)
        # render based on the render mode
        if self.render_mode == self.RenderMode.NO_GUI_OR_RENDERING:
            # we never want to render anything here (this is for complete headless mode)
            pass
        elif self.render_mode == self.RenderMode.NO_RENDERING:
            # throttle the rendering frequency to keep the UI responsive
            self._render_throttle_counter += 1
            if self._render_throttle_counter % self._render_throttle_period == 0:
                self._render_throttle_counter = 0
                # here we don't render viewport so don't need to flush fabric data
                # note: we don't call super().render() anymore because they do flush the fabric data
                self.set_setting("/app/player/playSimulations", False)
                self._app.update()
                self.set_setting("/app/player/playSimulations", True)
        else:
            # manually flush the fabric data to update Hydra textures
            _t1 = _t.perf_counter()
            self.forward()
            # render the simulation
            _t2 = _t.perf_counter()
            # note: we don't call super().render() anymore because they do above operation inside
            #  and we don't want to do it twice. We may remove it once we drop support for Isaac Sim 2022.2.
            self.set_setting("/app/player/playSimulations", False)
            self._app.update()
            self.set_setting("/app/player/playSimulations", True)
        # Throttle render() logging to every 50th call to avoid log flood
        if not hasattr(self, "_render_log_count"):
            self._render_log_count = 0
        self._render_log_count += 1
        if self._render_log_count <= 3 or self._render_log_count % 50 == 0:
            elapsed = _t.perf_counter() - _t0
            logger.debug(
                "[PERF][simulation_context] render() total took %s s (call #%s)",
                round(elapsed, 3), self._render_log_count,
            )

        # app.update() may be changing the cuda device, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

    """
    Operations - Override (extension)
    """

    async def reset_async(self, soft: bool = False):
        # need to load all "physics" information from the USD file
        if not soft:
            omni.physx.acquire_physx_interface().force_load_physics_from_usd()
        # play the simulation
        await super().reset_async(soft=soft)

    """
    Initialization/Destruction - Override.
    """

    def _init_stage(self, *args, **kwargs) -> Usd.Stage:
        _ = super()._init_stage(*args, **kwargs)
        with sim_utils.use_stage(self.get_initial_stage()):
            # a stage update here is needed for the case when physics_dt != rendering_dt, otherwise the app crashes
            # when in headless mode
            self.set_setting("/app/player/playSimulations", False)
            self._app.update()
            self.set_setting("/app/player/playSimulations", True)
            # set additional physx parameters and bind material
            self._set_additional_physx_params()
            # load flatcache/fabric interface
            self._load_fabric_interface()
            # return the stage
            return self.stage

    async def _initialize_stage_async(self, *args, **kwargs) -> Usd.Stage:
        await super()._initialize_stage_async(*args, **kwargs)
        # set additional physx parameters and bind material
        self._set_additional_physx_params()
        # load flatcache/fabric interface
        self._load_fabric_interface()
        # return the stage
        return self.stage

    @classmethod
    def instance(cls) -> SimulationContext | None:
        """Get the singleton instance, or None if not created."""
        return getattr(cls, "_instance", None)

    def __init__(self, cfg: SimulationCfg | None = None):
        """Initialize the simulation context.

        Args:
            cfg: Simulation configuration. Defaults to None (uses default config).
        """
        if getattr(type(self), "_instance", None) is not None:
            return  # Already initialized

        # Store config
        self.cfg = SimulationCfg() if cfg is None else cfg

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
        if sim_utils.has_kit():
            import omni.usd

            kit_context = omni.usd.get_context()
            if kit_context is not None and kit_context.get_stage() is not self.stage:
                kit_context.attach_stage_with_callback(stage_cache.GetId(self.stage).ToLongInt())

        # Acquire settings interface (SettingsManager: standalone dict or Omniverse when available)
        self.settings = SettingsManager.instance()
        self._settings_helper = SettingsHelper(self.settings)

        # Initialize USD physics scene and physics manager
        self._init_usd_physics_scene()
        # Set default physics backend if not specified
        if self.cfg.physics is None:
            from isaaclab_physx.physics import PhysxCfg

            self.cfg.physics = PhysxCfg()
        self._physics = self.cfg.physics
        self.physics_manager: type[PhysicsManager] = self._physics.class_type
        self.physics_manager.initialize(self)
        self._apply_render_cfg_settings()

        # Initialize visualizer state (provider/visualizers are created lazily during initialize_visualizers()).
        self._scene_data_provider: SceneDataProvider | None = None
        self._visualizers: list[Visualizer] = []
        self._visualizer_step_counter = 0
        # Default visualization dt used before/without visualizer initialization.
        physics_dt = getattr(self.cfg.physics, "dt", None)
        self._viz_dt = (physics_dt if physics_dt is not None else self.cfg.dt) * self.cfg.render_interval

        # Cache commonly-used settings (these don't change during runtime)
        self._has_gui = bool(self.get_setting("/isaaclab/has_gui"))
        self._has_offscreen_render = bool(self.get_setting("/isaaclab/render/offscreen"))
        # Note: has_rtx_sensors is NOT cached because it changes when Camera sensors are created

        # Simulation state
        self._is_playing = False
        self._is_stopped = True

        # Monotonic physics-step counter used by camera sensors for
        self._physics_step_count: int = 0

        setattr(type(self), "_instance", self)  # Mark as valid singleton only after successful init

    def get_initial_stage(self) -> Usd.Stage:
        """Return the current USD stage used by this context."""
        return self.stage

    def _apply_render_cfg_settings(self) -> None:
        """Apply render preset and overrides from SimulationCfg.render."""
        # TODO: Refactor render preset + override handling to a dedicated RenderingQualityCfg
        # (name subject to change) to keep quality profiles and carb mappings centralized.
        render_cfg = getattr(self.cfg, "render", None)
        if render_cfg is None:
            return

        # Priority:
        # 1) CLI/AppLauncher setting if present, 2) SimulationCfg.render.rendering_mode.
        rendering_mode = self.get_setting("/isaaclab/rendering/rendering_mode")
        if not rendering_mode:
            rendering_mode = getattr(render_cfg, "rendering_mode", None)

        if rendering_mode:
            supported_rendering_modes = {"performance", "balanced", "quality"}
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes "
                    f"{sorted(supported_rendering_modes)}."
                )

            isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
            from isaaclab.utils.version import get_isaac_sim_version

            if get_isaac_sim_version().major < 6:
                isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_5")

            preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
            if os.path.exists(preset_filename):
                with open(preset_filename) as file:
                    preset_dict = toml.load(file)

                def _apply_nested(data: dict[str, Any], path: str = "") -> None:
                    for key, value in data.items():
                        key_path = f"{path}/{key}" if path else f"/{key}"
                        if isinstance(value, dict):
                            _apply_nested(value, key_path)
                        else:
                            self.set_setting(key_path.replace(".", "/"), value)

                _apply_nested(preset_dict)
            else:
                logger.warning("[SimulationContext] Render preset file not found: %s", preset_filename)

        # RenderCfg fields mapped to setting paths (stored via SettingsManager)
        field_to_setting = {
            "enable_translucency": "/rtx/translucency/enabled",
            "enable_reflections": "/rtx/reflections/enabled",
            "enable_global_illumination": "/rtx/indirectDiffuse/enabled",
            "enable_dlssg": "/rtx-transient/dlssg/enabled",
            "enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
            "dlss_mode": "/rtx/post/dlss/execMode",
            "enable_direct_lighting": "/rtx/directLighting/enabled",
            "samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
            "enable_shadows": "/rtx/shadows/enabled",
            "enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
            "dome_light_upper_lower_strategy": "/rtx/domeLight/upperLowerStrategy",
        }

        for key, value in vars(render_cfg).items():
            if value is None or key in {"rendering_mode", "carb_settings", "antialiasing_mode"}:
                continue
            setting_path = field_to_setting.get(key)
            if setting_path is not None:
                self.set_setting(setting_path, value)

        # Raw overrides from render_cfg (stored via SettingsManager)
        extra_settings = getattr(render_cfg, "carb_settings", None)
        if extra_settings:
            for key, value in extra_settings.items():
                if "_" in key:
                    path = "/" + key.replace("_", "/")
                elif "." in key:
                    path = "/" + key.replace(".", "/")
                else:
                    path = key
                self.set_setting(path, value)

        # Optional anti-aliasing mode via Replicator (best-effort, may use Omniverse APIs)
        antialiasing_mode = getattr(render_cfg, "antialiasing_mode", None)
        if antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=antialiasing_mode)
            except Exception:
                pass

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
        """Returns whether rendering is active (GUI, RTX sensors, or visualizers requested)."""
        return (
            self._has_gui
            or self._has_offscreen_render
            or self.get_setting("/isaaclab/render/rtx_sensors")
            or bool(self.get_setting("/isaaclab/visualizer"))
        )

    def get_physics_dt(self) -> float:
        """Returns the physics time step."""
        return self.physics_manager.get_physics_dt()

    def _create_default_visualizer_configs(self, requested_visualizers: list[str]) -> list:
        """Create default visualizer configs for requested types."""
        default_configs = []
        for viz_type in requested_visualizers:
            try:
                if viz_type == "newton":
                    default_configs.append(NewtonVisualizerCfg())
                elif viz_type == "rerun":
                    default_configs.append(RerunVisualizerCfg())
                elif viz_type == "kit":
                    default_configs.append(KitVisualizerCfg())
                else:
                    logger.warning(
                        f"[SimulationContext] Unknown visualizer type '{viz_type}' requested. "
                        f"Valid types: {', '.join(repr(t) for t in _VISUALIZER_TYPES)}. Skipping."
                    )
            except Exception as exc:
                logger.error(f"[SimulationContext] Failed to create default config for visualizer '{viz_type}': {exc}")
        return default_configs

    def _get_cli_visualizer_types(self) -> list[str]:
        """Return list of visualizer types requested via CLI (setting)."""
        requested = self.get_setting("/isaaclab/visualizer")
        if not requested:
            return []
        parts = [p.strip() for p in requested.split(",") if p.strip()]
        return [v for part in parts for v in part.split() if v]

    def resolve_visualizer_types(self) -> list[str]:
        """Resolve visualizer types from config or CLI settings."""
        visualizer_cfgs = self.cfg.visualizer_cfgs
        if visualizer_cfgs is None:
            return self._get_cli_visualizer_types()

        if not isinstance(visualizer_cfgs, list):
            visualizer_cfgs = [visualizer_cfgs]
        return [cfg.visualizer_type for cfg in visualizer_cfgs if getattr(cfg, "visualizer_type", None)]

    def _resolve_visualizer_cfgs(self) -> list[Any]:
        """Resolve final visualizer configs from cfg and optional CLI override."""
        visualizer_cfgs: list[Any] = []
        if self.cfg.visualizer_cfgs is not None:
            visualizer_cfgs = (
                self.cfg.visualizer_cfgs if isinstance(self.cfg.visualizer_cfgs, list) else [self.cfg.visualizer_cfgs]
            )

        cli_requested = self._get_cli_visualizer_types()
        if not visualizer_cfgs:
            return self._create_default_visualizer_configs(cli_requested) if cli_requested else []

        if not cli_requested:
            return visualizer_cfgs

        # CLI selection is explicit: keep only requested cfg types, then add defaults for missing requested types.
        cli_requested_set = set(cli_requested)
        selected_cfgs = [cfg for cfg in visualizer_cfgs if getattr(cfg, "visualizer_type", None) in cli_requested_set]
        existing_types = {getattr(cfg, "visualizer_type", None) for cfg in selected_cfgs}
        for viz_type in cli_requested:
            if viz_type not in existing_types and viz_type in _VISUALIZER_TYPES:
                selected_cfgs.extend(self._create_default_visualizer_configs([viz_type]))
                existing_types.add(viz_type)
        return selected_cfgs

    def initialize_visualizers(self) -> None:
        """Initialize visualizers from SimulationCfg.visualizer_cfgs."""
        if self._visualizers:
            return

        physics_dt = getattr(self.cfg.physics, "dt", None)
        self._viz_dt = (physics_dt if physics_dt is not None else self.cfg.dt) * self.cfg.render_interval

        visualizer_cfgs = self._resolve_visualizer_cfgs()
        if not visualizer_cfgs:
            return

        self.initialize_scene_data_provider(visualizer_cfgs)
        self._visualizers = []

        for cfg in visualizer_cfgs:
            try:
                visualizer = cfg.create_visualizer()
                visualizer.initialize(self._scene_data_provider)
                self._visualizers.append(visualizer)
                logger.info(f"Initialized visualizer: {type(visualizer).__name__} (type: {cfg.visualizer_type})")
            except Exception as exc:
                logger.error(f"Failed to initialize visualizer '{cfg.visualizer_type}' ({type(cfg).__name__}): {exc}")

        if not self._visualizers and self._scene_data_provider is not None:
            close_provider = getattr(self._scene_data_provider, "close", None)
            if callable(close_provider):
                close_provider()
            self._scene_data_provider = None

    def initialize_scene_data_provider(self, visualizer_cfgs: list[Any]) -> SceneDataProvider:
        if self._scene_data_provider is None:
            from .scene_data_providers import PhysxSceneDataProvider

            # TODO: When Newton/Warp backend scene data provider is implemented and validated,
            # switch provider selection to route by physics backend:
            # - Omni/PhysX -> PhysxSceneDataProvider
            # - Newton/Warp -> NewtonSceneDataProvider
            self._scene_data_provider = PhysxSceneDataProvider(visualizer_cfgs, self.stage, self)
        return self._scene_data_provider

    @property
    def visualizers(self) -> list[Visualizer]:
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
        if not self._visualizers:
            # Initialize visualizers after PhysX sim view is ready.
            self.initialize_visualizers()
        self._is_playing = True
        self._is_stopped = False

    def step(self, render: bool = True) -> None:
        """Step physics and optionally render.

        Args:
            render: Whether to render the scene after stepping. Defaults to True.
        """
        self._physics_step_count += 1
        self.physics_manager.step()
        if render and self.is_rendering:
            self.render()

    def render(self, mode: int | None = None) -> None:
        """Update visualizers and render the scene.

        Calls update_visualizers() so visualizers run at the render cadence (not at
        every physics step). Camera sensors drive their configured renderer when
        fetching data, so this method remains backend-agnostic.
        """
        self.update_visualizers(self.get_rendering_dt())

        # Call render callbacks
        if hasattr(self, "_render_callbacks"):
            for callback in self._render_callbacks.values():
                callback(None)  # Pass None as event data

    def update_visualizers(self, dt: float) -> None:
        """Update visualizers without triggering renderer/GUI."""
        if not self._visualizers:
            return

        self.update_scene_data_provider()

        visualizers_to_remove = []
        for viz in self._visualizers:
            try:
                if viz.is_rendering_paused():
                    continue
                if viz.is_closed:
                    logger.info("Visualizer closed: %s", type(viz).__name__)
                    visualizers_to_remove.append(viz)
                    continue
                if not viz.is_running():
                    logger.info("Visualizer not running: %s", type(viz).__name__)
                    visualizers_to_remove.append(viz)
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

    def update_scene_data_provider(self, force_require_forward: bool = False):
        if force_require_forward or self._should_forward_before_visualizer_update():
            self.physics_manager.forward()
        self._visualizer_step_counter += 1
        if self._scene_data_provider is None:
            return
        provider = self._scene_data_provider
        env_ids_union: list[int] = []
        for viz in self._visualizers:
            ids = viz.get_visualized_env_ids()
            if ids is not None:
                env_ids_union.extend(ids)
        env_ids = list(dict.fromkeys(env_ids_union)) if env_ids_union else None
        provider.update(env_ids)

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

    @classmethod
    def clear_instance(cls) -> None:
        """Clean up resources and clear the singleton instance."""
        inst = getattr(cls, "_instance", None)
        if inst is not None:
            # Close physics manager FIRST to detach PhysX from the stage
            # This must happen before clearing USD prims to avoid PhysX cleanup errors
            inst.physics_manager.close()

            # Now safe to clear stage contents (PhysX is detached)
            cls.clear_stage()

            # Close all visualizers
            for viz in inst._visualizers:
                viz.close()
            inst._visualizers.clear()
            if inst._scene_data_provider is not None:
                close_provider = getattr(inst._scene_data_provider, "close", None)
                if callable(close_provider):
                    close_provider()
                inst._scene_data_provider = None

            # Close the stage (clears cache, thread-local context, and Kit USD context)
            stage_utils.close_stage()

            # Clear instance
            setattr(cls, "_instance", None)

            gc.collect()
            logger.info("SimulationContext cleared")

    @classmethod
    def clear_stage(cls) -> None:
        """Clear the current USD stage (preserving /World and PhysicsScene).

        Uses a predicate that preserves /World and PhysicsScene while also
        respecting the default deletability checks (ancestral prims, etc.).
        """
        if getattr(cls, "_instance", None) is None:
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
    sim: SimulationContext | None = None
    try:
        if create_new_stage:
            sim_utils.create_new_stage()

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
        if sim is not None:
            if not sim.get_setting("/isaaclab/has_gui"):
                sim.stop()
            sim.clear_instance()
