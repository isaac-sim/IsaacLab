# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import enum
import glob
import numpy as np
import os
import re
import time
import toml
import torch
import traceback
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import carb
import flatdict
import isaacsim.core.utils.stage as stage_utils
import omni.log
import omni.physx
import omni.usd
from isaacsim.core.api.simulation_context import SimulationContext as _SimulationContext
from isaacsim.core.utils.carb import get_carb_setting, set_carb_setting
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.version import get_version
from pxr import Gf, PhysxSchema, Usd, UsdPhysics

from isaaclab.sim.utils import create_new_stage_in_memory, use_stage

from .simulation_cfg import SimulationCfg
from .spawners import DomeLightCfg, GroundPlaneCfg
from .utils import bind_physics_material


class SimulationContext(_SimulationContext):
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
        # store input
        if cfg is None:
            cfg = SimulationCfg()
        # check that the config is valid
        cfg.validate()
        self.cfg = cfg
        # check that simulation is running
        if stage_utils.get_current_stage() is None:
            raise RuntimeError("The stage has not been created. Did you run the simulator?")

        # create stage in memory if requested
        if self.cfg.create_stage_in_memory:
            self._initial_stage = create_new_stage_in_memory()
        else:
            self._initial_stage = omni.usd.get_context().get_stage()

        # acquire settings interface
        self.carb_settings = carb.settings.get_settings()

        # read isaac sim version (this includes build tag, release tag etc.)
        # note: we do it once here because it reads the VERSION file from disk and is not expected to change.
        self._isaacsim_version = get_version()

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
            omni.log.warn(
                "Large simulation step size (> 0.0333 seconds) is not recommended without enabling stabilization."
                " Consider setting the `enable_stabilization` flag to True in the PhysxCfg, or reducing the"
                " simulation step size if you run into physics issues."
            )

        # create a simulation context to control the simulator
        if float(".".join(self._isaacsim_version[2])) < 5:
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

    def _apply_physics_settings(self):
        """Sets various carb physics settings."""
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        set_carb_setting(self.carb_settings, "/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        set_carb_setting(self.carb_settings, "/physics/physxDispatcher", True)
        # disable contact processing in omni.physx
        # note: we disable it by default to avoid the overhead of contact processing when it isn't needed.
        #   The physics flag gets enabled when a contact sensor is created.
        if hasattr(self.cfg, "disable_contact_processing"):
            omni.log.warn(
                "The `disable_contact_processing` attribute is deprecated and always set to True"
                " to avoid unnecessary overhead. Contact processing is automatically enabled when"
                " a contact sensor is created, so manual configuration is no longer required."
            )
        # FIXME: From investigation, it seems this flag only affects CPU physics. For GPU physics, contacts
        #  are always processed. The issue is reported to the PhysX team by @mmittal.
        set_carb_setting(self.carb_settings, "/physics/disableContactProcessing", True)
        # disable custom geometry for cylinder and cone collision shapes to allow contact reporting for them
        # reason: cylinders and cones aren't natively supported by PhysX so we need to use custom geometry flags
        # reference: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html?highlight=capsule#geometry
        set_carb_setting(self.carb_settings, "/physics/collisionConeCustomGeometry", False)
        set_carb_setting(self.carb_settings, "/physics/collisionCylinderCustomGeometry", False)
        # hide the Simulation Settings window
        set_carb_setting(self.carb_settings, "/physics/autoPopupSimulationOutputWindow", False)

    def _apply_render_settings_from_cfg(self):
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
        }

        not_carb_settings = ["rendering_mode", "carb_settings", "antialiasing_mode"]

        # grab the rendering mode using the following priority:
        # 1. command line argument --rendering_mode, if provided
        # 2. rendering_mode from Render Config, if set
        # 3. lastly, default to "balanced" mode, if neither is specified
        rendering_mode = get_carb_setting(self.carb_settings, "/isaaclab/rendering/rendering_mode")
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
            if float(".".join(self._isaacsim_version[2])) < 5:
                isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_4_5")

            # grab preset settings
            preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
            with open(preset_filename) as file:
                preset_dict = toml.load(file)
            preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

            # set presets
            for key, value in preset_dict.items():
                key = "/" + key.replace(".", "/")  # convert to carb setting format
                set_carb_setting(self.carb_settings, key, value)

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
            set_carb_setting(self.carb_settings, key, value)

        # set general carb settings
        carb_settings = self.cfg.render.carb_settings
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")  # convert from python variable style string
                elif "." in key:
                    key = "/" + key.replace(".", "/")  # convert from .kit file style string
                if get_carb_setting(self.carb_settings, key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.general_parameters does not map to a carb setting.")
                set_carb_setting(self.carb_settings, key, value)

        # set denoiser mode
        if self.cfg.render.antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=self.cfg.render.antialiasing_mode)
            except Exception:
                pass

        # WAR: Ensure /rtx/renderMode RaytracedLighting is correctly cased.
        if get_carb_setting(self.carb_settings, "/rtx/rendermode").lower() == "raytracedlighting":
            set_carb_setting(self.carb_settings, "/rtx/rendermode", "RaytracedLighting")

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

        This is a wrapper around the ``isaacsim.core.version.get_version()`` function.

        The returned tuple contains the following information:

        * Major version (int): This is the year of the release (e.g. 2022).
        * Minor version (int): This is the half-year of the release (e.g. 1 or 2).
        * Patch version (int): This is the patch number of the release (e.g. 0).
        """
        return int(self._isaacsim_version[2]), int(self._isaacsim_version[3]), int(self._isaacsim_version[4])

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
            omni.log.warn(
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
        self._settings.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Read the simulation setting using the Carbonite SDK.

        Args:
            name: The name of the setting.

        Returns:
            The value of the setting.
        """
        return self._settings.get(name)

    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        if self._fabric_iface is not None:
            if self.physics_sim_view is not None and self.is_playing():
                # Update the articulations' link's poses before rendering
                self.physics_sim_view.update_articulations_kinematic()
            self._update_fabric(0.0, 0.0)

    def get_initial_stage(self) -> Usd.Stage:
        """Returns stage handle used during scene creation.

        Returns:
            The stage used during scene creation.
        """
        return self._initial_stage

    """
    Operations - Override (standalone)
    """

    def reset(self, soft: bool = False):
        self._disable_app_control_on_stop_handle = True
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
        super().reset(soft=soft)
        # app.update() may be changing the cuda device in reset, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
        # enable kinematic rendering with fabric
        if self.physics_sim_view:
            self.physics_sim_view._backend.initialize_kinematic_bodies()
        # perform additional rendering steps to warm up replicator buffers
        # this is only needed for the first time we set the simulation
        if not soft:
            for _ in range(2):
                self.render()
        self._disable_app_control_on_stop_handle = False

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
                carb.log_warn("[INFO][SimulationContext]: Animation recording finished. Closing app.")
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
        super().step(render=render)

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
            self.forward()
            # render the simulation
            # note: we don't call super().render() anymore because they do above operation inside
            #  and we don't want to do it twice. We may remove it once we drop support for Isaac Sim 2022.2.
            self.set_setting("/app/player/playSimulations", False)
            self._app.update()
            self.set_setting("/app/player/playSimulations", True)

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
        with use_stage(self.get_initial_stage()):
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
    def clear_instance(cls):
        # clear the callback
        if cls._instance is not None:
            if cls._instance._app_control_on_stop_handle is not None:
                cls._instance._app_control_on_stop_handle.unsubscribe()
                cls._instance._app_control_on_stop_handle = None
        # call parent to clear the instance
        super().clear_instance()

    """
    Helper Functions
    """

    def _set_additional_physx_params(self):
        """Sets additional PhysX parameters that are not directly supported by the parent class."""
        # obtain the physics scene api
        physics_scene: UsdPhysics.Scene = self._physics_context._physics_scene
        physx_scene_api: PhysxSchema.PhysxSceneAPI = self._physics_context._physx_scene_api
        # assert that scene api is not None
        if physx_scene_api is None:
            raise RuntimeError("Physics scene API is None! Please create the scene first.")
        # set parameters not directly supported by the constructor
        # -- Continuous Collision Detection (CCD)
        # ref: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html?highlight=ccd#continuous-collision-detection
        self._physics_context.enable_ccd(self.cfg.physx.enable_ccd)
        # -- GPU collision stack size
        physx_scene_api.CreateGpuCollisionStackSizeAttr(self.cfg.physx.gpu_collision_stack_size)
        # -- Improved determinism by PhysX
        physx_scene_api.CreateEnableEnhancedDeterminismAttr(self.cfg.physx.enable_enhanced_determinism)

        # -- Gravity
        # note: Isaac sim only takes the "up-axis" as the gravity direction. But physics allows any direction so we
        #  need to convert the gravity vector to a direction and magnitude pair explicitly.
        gravity = np.asarray(self.cfg.gravity)
        gravity_magnitude = np.linalg.norm(gravity)

        # Avoid division by zero
        if gravity_magnitude != 0.0:
            gravity_direction = gravity / gravity_magnitude
        else:
            gravity_direction = gravity

        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
        physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)

        # position iteration count
        physx_scene_api.CreateMinPositionIterationCountAttr(self.cfg.physx.min_position_iteration_count)
        physx_scene_api.CreateMaxPositionIterationCountAttr(self.cfg.physx.max_position_iteration_count)
        # velocity iteration count
        physx_scene_api.CreateMinVelocityIterationCountAttr(self.cfg.physx.min_velocity_iteration_count)
        physx_scene_api.CreateMaxVelocityIterationCountAttr(self.cfg.physx.max_velocity_iteration_count)

        # create the default physics material
        # this material is used when no material is specified for a primitive
        # check: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-materials
        material_path = f"{self.cfg.physics_prim_path}/defaultMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        # bind the physics material to the scene
        bind_physics_material(self.cfg.physics_prim_path, material_path)

    def _load_fabric_interface(self):
        """Loads the fabric interface if enabled."""
        if self.cfg.use_fabric:
            from omni.physxfabric import get_physx_fabric_interface

            # acquire fabric interface
            self._fabric_iface = get_physx_fabric_interface()
            if hasattr(self._fabric_iface, "force_update"):
                # The update method in the fabric interface only performs an update if a physics step has occurred.
                # However, for rendering, we need to force an update since any element of the scene might have been
                # modified in a reset (which occurs after the physics step) and we want the renderer to be aware of
                # these changes.
                self._update_fabric = self._fabric_iface.force_update
            else:
                # Needed for backward compatibility with older Isaac Sim versions
                self._update_fabric = self._fabric_iface.update

    def _update_anim_recording(self):
        """Tracks anim recording timestamps and triggers finish animation recording if the total time has elapsed."""
        if self._anim_recording_started_timestamp is None:
            self._anim_recording_started_timestamp = time.time()

        if self._anim_recording_started_timestamp is not None:
            anim_recording_total_time = time.time() - self._anim_recording_started_timestamp
            if anim_recording_total_time > self._anim_recording_stop_time:
                self._finish_anim_recording()
                return True
        return False

    def _setup_anim_recording(self):
        """Sets up anim recording settings and initializes the recording."""

        self._anim_recording_enabled = bool(self.carb_settings.get("/isaaclab/anim_recording/enabled"))
        if not self._anim_recording_enabled:
            return

        # Import omni.physx.pvd.bindings here since it is not available by default
        from omni.physxpvd.bindings import _physxPvd

        # Init anim recording settings
        self._anim_recording_start_time = self.carb_settings.get("/isaaclab/anim_recording/start_time")
        self._anim_recording_stop_time = self.carb_settings.get("/isaaclab/anim_recording/stop_time")
        self._anim_recording_first_step_timestamp = None
        self._anim_recording_started_timestamp = None

        # Make output path relative to repo path
        repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
        self._anim_recording_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self._anim_recording_output_dir = (
            os.path.join(repo_path, "anim_recordings", self._anim_recording_timestamp).replace("\\", "/").rstrip("/")
            + "/"
        )
        os.makedirs(self._anim_recording_output_dir, exist_ok=True)

        # Acquire physx pvd interface and set output directory
        self._physxPvdInterface = _physxPvd.acquire_physx_pvd_interface()

        # Set carb settings for the output path and enabling pvd recording
        set_carb_setting(
            self.carb_settings, "/persistent/physics/omniPvdOvdRecordingDirectory", self._anim_recording_output_dir
        )
        set_carb_setting(self.carb_settings, "/physics/omniPvdOutputEnabled", True)

    def _update_usda_start_time(self, file_path, start_time):
        """Updates the start time of the USDA baked anim recordingfile."""

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

    def _finish_anim_recording(self):
        """Finishes the animation recording and outputs the baked animation recording."""

        carb.log_warn(
            "[INFO][SimulationContext]: Finishing animation recording. Stage must be saved. Might take a few minutes."
        )

        # Detaching the stage will also close it and force the serialization of the OVD file
        physx = omni.physx.get_physx_simulation_interface()
        physx.detach_stage()

        # Save stage to disk
        stage_path = os.path.join(self._anim_recording_output_dir, "stage_simulation.usdc")
        stage_utils.save_stage(stage_path, save_and_reload_in_place=False)

        # Find the latest ovd file not named tmp.ovd
        ovd_files = [
            f for f in glob.glob(os.path.join(self._anim_recording_output_dir, "*.ovd")) if not f.endswith("tmp.ovd")
        ]
        input_ovd_path = max(ovd_files, key=os.path.getctime)

        # Invoke pvd interface to create recording
        stage_filename = "baked_animation_recording.usda"
        result = self._physxPvdInterface.ovd_to_usd_over_with_layer_creation(
            input_ovd_path,
            stage_path,
            self._anim_recording_output_dir,
            stage_filename,
            self._anim_recording_start_time,
            self._anim_recording_stop_time,
            True,  # True: ASCII layers / False : USDC layers
            False,  # True: verify over layer
        )

        # Workaround for manually setting the truncated start time in the baked animation recording
        self._update_usda_start_time(
            os.path.join(self._anim_recording_output_dir, stage_filename), self._anim_recording_start_time
        )

        # Disable recording
        set_carb_setting(self.carb_settings, "/physics/omniPvdOutputEnabled", False)

        return result

    """
    Callbacks.
    """

    def _app_control_on_stop_handle_fn(self, event: carb.events.IEvent):
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
        if not self._disable_app_control_on_stop_handle:
            while not omni.timeline.get_timeline_interface().is_playing():
                self.render()
        return


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

        if add_lighting or (auto_add_lighting and sim.has_gui()):
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
        omni.log.error(traceback.format_exc())
        raise
    finally:
        if not sim.has_gui():
            # Stop simulation only if we aren't rendering otherwise the app will hang indefinitely
            sim.stop()

        # Clear the stage
        sim.clear_all_callbacks()
        sim.clear_instance()
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
