# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import enum
import flatdict
import logging
import os
import toml
import torch
import traceback
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import carb
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd, UsdUtils

import isaaclab.sim as sim_utils
from isaaclab.utils.logger import configure_logging
from isaaclab.utils.version import get_isaac_sim_version

from .physx_backend import PhysXBackend
from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg

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

        * :attr:`NO_GUI_OR_RENDERING`: The simulation is running without a GUI and off-screen rendering flag is disabled,
          so none of the above are updated.
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

        # note: we read this once since it is not expected to change during runtime
        # read flag for whether a local GUI is enabled
        self._local_gui = self.carb_settings.get("/app/window/enabled")
        # read flag for whether livestreaming GUI is enabled
        self._livestream_gui = self.carb_settings.get("/app/livestream/enabled")
        # read flag for whether XR GUI is enabled
        self._xr_gui = self.carb_settings.get("/app/xr/enabled")

        # read flag for whether the Isaac Lab viewport capture pipeline will be used,
        # casting None to False if the flag doesn't exist
        # this flag is set from the AppLauncher class
        self._offscreen_render = bool(self.carb_settings.get("/isaaclab/render/offscreen"))
        # read flag for whether the default viewport should be enabled
        self._render_viewport = bool(self.carb_settings.get("/isaaclab/render/active_viewport"))
        # flag for whether any GUI will be rendered (local, livestreamed or viewport)
        has_gui = self._local_gui or self._livestream_gui or self._xr_gui
        self.carb_settings.set_bool("/isaaclab/has_gui", has_gui)
        # apply render settings from render config
        self._apply_render_settings_from_cfg()

        # store the default render mode
        if not self.carb_settings.get("/isaaclab/has_gui") and not self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.NO_GUI_OR_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        elif not self.carb_settings.get("/isaaclab/has_gui") and self._offscreen_render:
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
        if self.carb_settings.get("/isaaclab/has_gui"):
            self.cfg.enable_scene_query_support = True

        # create a tensor for gravity
        # note: this line is needed to create a "tensor" in the device to avoid issues with torch 2.1 onwards.
        #   the issue is with some heap memory corruption when torch tensor is created inside the asset class.
        #   you can reproduce the issue by commenting out this line and running the test `test_articulation.py`.
        self._gravity_tensor = torch.tensor(self.cfg.gravity, dtype=torch.float32, device=self.cfg.device)

        # define a global variable to store the exceptions raised in the callback stack
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None

        # add callback to deal the simulation app when simulation is stopped.
        # this is needed because physics views go invalid once we stop the simulation
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._app_control_on_stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda *args, obj=weakref.proxy(self): obj._app_control_on_stop_handle_fn(*args),
            order=15,
        )
        self._disable_app_control_on_stop_handle = False

        # obtain interfaces for simulation
        self._app_iface = omni.kit.app.get_app_interface()
        self._timeline_iface = omni.timeline.get_timeline_interface()

        # set timeline auto update to True
        self._timeline_iface.set_auto_update(True)

        # initialize physics backend (handles scene creation, settings, fabric)
        self._physics_backend = PhysXBackend(self)

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
            # clear the callback
            if cls._instance._app_control_on_stop_handle is not None:
                cls._instance._app_control_on_stop_handle.unsubscribe()
                cls._instance._app_control_on_stop_handle = None
            # close physics backend (clears SimulationManager, detaches physx stage)
            cls._instance._physics_backend.close()
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
    def app(self) -> omni.kit.app.IApp:
        """Omniverse Kit Application interface."""
        return self._app_iface

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
        return self._physics_backend.device

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
        return self.cfg.dt

    @property
    def physics_prim_path(self) -> str:
        """The path to the physics scene prim."""
        return self.cfg.physics_prim_path

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step of the simulation.

        Returns:
            The rendering time step of the simulation.
        """
        return self.cfg.dt * self.cfg.render_interval

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
        # safe call only if we have a GUI or viewport rendering enabled
        if self.carb_settings.get("/isaaclab/has_gui") or self._offscreen_render or self._render_viewport:
            set_camera_view(eye, target, camera_prim_path)

    def set_render_mode(self, mode: RenderMode):
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the different render modes.

        .. note::
            When no GUI is available (locally or livestreamed), we do not need to choose whether the viewport
            needs to render or not (since there is no GUI). Thus, in this case, calling the function will not
            change the render mode.

        Args:
            mode: The rendering mode. If different than SimulationContext's rendering mode,
                SimulationContext's mode is changed to the new mode.

        Raises:
            ValueError: If the input mode is not supported.
        """
        # check if mode change is possible -- not possible when no GUI is available
        if not self.carb_settings.get("/isaaclab/has_gui"):
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
        return self._timeline_iface.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped.

        Returns:
            True if the simulator is stopped.
        """
        return self._timeline_iface.is_stopped()

    def play(self) -> None:
        """Start playing the simulation."""
        # play the simulation
        self._timeline_iface.play()
        self._timeline_iface.commit()
        # perform one step to propagate all physics handles properly
        self.set_setting("/app/player/playSimulations", False)
        self._app_iface.update()
        self.set_setting("/app/player/playSimulations", True)
        # check for callback exceptions
        self._check_for_callback_exceptions()

    def pause(self) -> None:
        """Pause the simulation."""
        # pause the simulation
        self._timeline_iface.pause()
        self._timeline_iface.commit()
        # we don't need to propagate physics handles since no callbacks are triggered during pause
        # check for callback exceptions
        self._check_for_callback_exceptions()

    def stop(self) -> None:
        """Stop the simulation.

        Note:
            Stopping the simulation will lead to the simulation state being lost.
        """
        # stop the simulation
        self._timeline_iface.stop()
        self._timeline_iface.commit()
        # perform one step to propagate all physics handles properly
        self.set_setting("/app/player/playSimulations", False)
        self._app_iface.update()
        self.set_setting("/app/player/playSimulations", True)
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
        # reset the simulation
        if not soft:
            # disable app control on stop handle
            self._disable_app_control_on_stop_handle = True
            if not self.is_stopped():
                self.stop()
            self._disable_app_control_on_stop_handle = False
            # play the simulation
            self.play()
            # check for callback exceptions
            self._check_for_callback_exceptions()

        # reset physics backend (initializes physics, resets cuda device, kinematic bodies)
        self._physics_backend.reset(soft)

        # perform additional rendering steps to warm up replicator buffers
        # this is only needed for the first time we set the simulation
        if not soft:
            for _ in range(2):
                self.render()

    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        self._physics_backend.forward()

    def step(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                    If set to False, the scene is not rendered and only the physics simulation is stepped.
        """
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
            self._app_iface.update()

        # step the physics simulation
        self._physics_backend.step(render)

    def render(self, mode: RenderMode | None = None):
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
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
                self.set_setting("/app/player/playSimulations", False)
                self._app_iface.update()
                self.set_setting("/app/player/playSimulations", True)
        else:
            # manually flush the fabric data to update Hydra textures
            self.forward()
            # render the simulation
            # note: we don't call super().render() anymore because they do above operation inside
            #  and we don't want to do it twice. We may remove it once we drop support for Isaac Sim 2022.2.
            self.set_setting("/app/player/playSimulations", False)
            self._app_iface.update()
            self.set_setting("/app/player/playSimulations", True)

        # app.update() may be changing the cuda device, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

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

    def _apply_render_settings_from_cfg(self):  # noqa: C901
        """Sets rtx settings specified in the RenderCfg."""

        # define mapping of user-friendly RenderCfg names to native carb names
        rendering_setting_name_mapping = {
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

        not_carb_settings = ["rendering_mode", "carb_settings", "antialiasing_mode"]

        # grab the rendering mode using the following priority:
        # 1. command line argument --rendering_mode, if provided
        # 2. rendering_mode from Render Config, if set
        # 3. lastly, default to "balanced" mode, if neither is specified
        rendering_mode = self.carb_settings.get("/isaaclab/rendering/rendering_mode")
        if not rendering_mode:
            rendering_mode = self.cfg.render.rendering_mode
        if not rendering_mode:
            rendering_mode = "balanced"

        # set preset settings (same behavior as the CLI arg --rendering_mode)
        if rendering_mode is not None:
            # check if preset is supported
            supported_rendering_modes = ["performance", "balanced", "quality"]
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes {supported_rendering_modes}."
                )

            # grab isaac lab apps path
            isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
            # for Isaac Sim 4.5 compatibility, we use the 4.5 rendering mode app files in a different folder
            if get_isaac_sim_version().major < 5:
                isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_4_5")

            # grab preset settings
            preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
            with open(preset_filename) as file:
                preset_dict = toml.load(file)
            preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

            # set presets
            for key, value in preset_dict.items():
                key = "/" + key.replace(".", "/")  # convert to carb setting format
                self.set_setting(key, value)

        # set user-friendly named settings
        for key, value in vars(self.cfg.render).items():
            if value is None or key in not_carb_settings:
                # skip unset settings and non-carb settings
                continue
            if key not in rendering_setting_name_mapping:
                raise ValueError(
                    f"'{key}' in RenderCfg not found. Note: internal 'rendering_setting_name_mapping' dictionary might"
                    " need to be updated."
                )
            key = rendering_setting_name_mapping[key]
            self.set_setting(key, value)

        # set general carb settings
        carb_settings = self.cfg.render.carb_settings
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")  # convert from python variable style string
                elif "." in key:
                    key = "/" + key.replace(".", "/")  # convert from .kit file style string
                if self.get_setting(key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.general_parameters does not map to a carb setting.")
                self.set_setting(key, value)

        # set denoiser mode
        if self.cfg.render.antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=self.cfg.render.antialiasing_mode)
            except Exception:
                pass

        # WAR: Ensure /rtx/renderMode RaytracedLighting is correctly cased.
        if self.carb_settings.get("/rtx/rendermode").lower() == "raytracedlighting":
            self.carb_settings.set_string("/rtx/rendermode", "RaytracedLighting")

    def _check_for_callback_exceptions(self):
        """Checks for callback exceptions and raises them if found."""
        # disable simulation stopping control so that we can crash the program
        # if an exception is raised in a callback.
        self._disable_app_control_on_stop_handle = True
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:  # type: ignore
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None  # type: ignore
            raise exception_to_raise
        # re-enable simulation stopping control
        self._disable_app_control_on_stop_handle = False

    """
    Callbacks.
    """

    def _app_control_on_stop_handle_fn(self, _):
        """Callback to deal with the app when the simulation is stopped.

        Once the simulation is stopped, the physics handles go invalid. After that, it is not possible to
        resume the simulation from the last state. This leaves the app in an inconsistent state, where
        two possible actions can be taken:

        1. **Keep the app rendering**: In this case, the simulation is kept running and the app is not shutdown.
           However, the physics is not updated and the script cannot be resumed from the last state. The
           user has to manually close the app to stop the simulation.
        2. **Shutdown the app**: This is the default behavior. In this case, the app is shutdown and
           the simulation is stopped.

        Note:
            This callback is used only when running the simulation in a standalone python script. In an extension,
            it is expected that the user handles the extension shutdown.
        """
        pass
        # if not self._disable_app_control_on_stop_handle:
        #     while not omni.timeline.get_timeline_interface().is_playing():
        #         self.render()


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
            # Construct one and overwrite the dt, gravity, and device
            sim_cfg = SimulationCfg(dt=dt)

            # Set up gravity
            if gravity_enabled:
                sim_cfg.gravity = (0.0, 0.0, -9.81)
            else:
                sim_cfg.gravity = (0.0, 0.0, 0.0)

            # Set device
            sim_cfg.device = device

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
