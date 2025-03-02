# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import enum
import numpy as np
import sys
import torch
import traceback
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.log
import omni.physx
from isaacsim.core.api.simulation_context import SimulationContext as _SimulationContext
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.version import get_version
from pxr import Gf, PhysxSchema, Usd, UsdPhysics

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

        # set flags for simulator
        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        carb_settings_iface.set_bool("/physics/physxDispatcher", True)
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
        carb_settings_iface.set_bool("/physics/disableContactProcessing", True)
        # disable custom geometry for cylinder and cone collision shapes to allow contact reporting for them
        # reason: cylinders and cones aren't natively supported by PhysX so we need to use custom geometry flags
        # reference: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html?highlight=capsule#geometry
        carb_settings_iface.set_bool("/physics/collisionConeCustomGeometry", False)
        carb_settings_iface.set_bool("/physics/collisionCylinderCustomGeometry", False)
        # hide the Simulation Settings window
        carb_settings_iface.set_bool("/physics/autoPopupSimulationOutputWindow", False)
        # note: we read this once since it is not expected to change during runtime
        # read flag for whether a local GUI is enabled
        self._local_gui = carb_settings_iface.get("/app/window/enabled")
        # read flag for whether livestreaming GUI is enabled
        self._livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

        # read flag for whether the Isaac Lab viewport capture pipeline will be used,
        # casting None to False if the flag doesn't exist
        # this flag is set from the AppLauncher class
        self._offscreen_render = bool(carb_settings_iface.get("/isaaclab/render/offscreen"))
        # read flag for whether the default viewport should be enabled
        self._render_viewport = bool(carb_settings_iface.get("/isaaclab/render/active_viewport"))
        # flag for whether any GUI will be rendered (local, livestreamed or viewport)
        self._has_gui = self._local_gui or self._livestream_gui

        # apply render settings from render config
        carb_settings_iface.set_bool("/rtx/translucency/enabled", self.cfg.render.enable_translucency)
        carb_settings_iface.set_bool("/rtx/reflections/enabled", self.cfg.render.enable_reflections)
        carb_settings_iface.set_bool("/rtx/indirectDiffuse/enabled", self.cfg.render.enable_global_illumination)
        carb_settings_iface.set_bool("/rtx-transient/dlssg/enabled", self.cfg.render.enable_dlssg)
        carb_settings_iface.set_bool("/rtx-transient/dldenoiser/enabled", self.cfg.render.enable_dl_denoiser)
        carb_settings_iface.set_int("/rtx/post/dlss/execMode", self.cfg.render.dlss_mode)
        carb_settings_iface.set_bool("/rtx/directLighting/enabled", self.cfg.render.enable_direct_lighting)
        carb_settings_iface.set_int(
            "/rtx/directLighting/sampledLighting/samplesPerPixel", self.cfg.render.samples_per_pixel
        )
        carb_settings_iface.set_bool("/rtx/shadows/enabled", self.cfg.render.enable_shadows)
        carb_settings_iface.set_bool("/rtx/ambientOcclusion/enabled", self.cfg.render.enable_ambient_occlusion)
        # set denoiser mode
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=self.cfg.render.antialiasing_mode)
        except Exception:
            pass

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
        # read isaac sim version (this includes build tag, release tag etc.)
        # note: we do it once here because it reads the VERSION file from disk and is not expected to change.
        self._isaacsim_version = get_version()

        # create a tensor for gravity
        # note: this line is needed to create a "tensor" in the device to avoid issues with torch 2.1 onwards.
        #   the issue is with some heap memory corruption when torch tensor is created inside the asset class.
        #   you can reproduce the issue by commenting out this line and running the test `test_articulation.py`.
        self._gravity_tensor = torch.tensor(self.cfg.gravity, dtype=torch.float32, device=self.cfg.device)

        # add callback to deal the simulation app when simulation is stopped.
        # this is needed because physics views go invalid once we stop the simulation
        if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
            self._app_control_on_stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP),
                lambda *args, obj=weakref.proxy(self): obj._app_control_on_stop_callback(*args),
                order=15,
            )
        else:
            self._app_control_on_stop_handle = None

        # flatten out the simulation dictionary
        sim_params = self.cfg.to_dict()
        if sim_params is not None:
            if "physx" in sim_params:
                physx_params = sim_params.pop("physx")
                sim_params.update(physx_params)
        # create a simulation context to control the simulator
        super().__init__(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg.dt,
            rendering_dt=self.cfg.dt * self.cfg.render_interval,
            backend="torch",
            sim_params=sim_params,
            physics_prim_path=self.cfg.physics_prim_path,
            device=self.cfg.device,
        )

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

    """
    Operations - Override (standalone)
    """

    def reset(self, soft: bool = False):
        super().reset(soft=soft)
        # enable kinematic rendering with fabric
        if self.physics_sim_view:
            self.physics_sim_view._backend.initialize_kinematic_bodies()
        # perform additional rendering steps to warm up replicator buffers
        # this is only needed for the first time we set the simulation
        if not soft:
            for _ in range(2):
                self.render()

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
            self.app.update()

        # step the simulation
        super().step(render=render)

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

    """
    Callbacks.
    """

    def _app_control_on_stop_callback(self, event: carb.events.IEvent):
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
        # check if the simulation is stopped
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # keep running the simulator when configured to not shutdown the app
            if self._has_gui and sys.exc_info()[0] is None:
                omni.log.warn(
                    "Simulation is stopped. The app will keep running with physics disabled."
                    " Press Ctrl+C or close the window to exit the app."
                )
                while self.app.is_running():
                    self.render()

        # Note: For the following code:
        #   The method is an exact copy of the implementation in the `isaacsim.simulation_app.SimulationApp` class.
        #   We need to remove this method once the SimulationApp class becomes a singleton.

        # make sure that any replicator workflows finish rendering/writing
        try:
            import omni.replicator.core as rep

            rep_status = rep.orchestrator.get_status()
            if rep_status not in [rep.orchestrator.Status.STOPPED, rep.orchestrator.Status.STOPPING]:
                rep.orchestrator.stop()
            if rep_status != rep.orchestrator.Status.STOPPED:
                rep.orchestrator.wait_until_complete()

            # Disable capture on play to avoid replicator engaging on any new timeline events
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass

        # clear the instance and all callbacks
        # note: clearing callbacks is important to prevent memory leaks
        self.clear_all_callbacks()

        # workaround for exit issues, clean the stage first:
        if omni.usd.get_context().can_close_stage():
            omni.usd.get_context().close_stage()

        # print logging information
        print("[INFO]: Simulation is stopped. Shutting down the app.")

        # Cleanup any running tracy instances so data is not lost
        try:
            profiler_tracy = carb.profiler.acquire_profiler_interface(plugin_name="carb.profiler-tracy.plugin")
            if profiler_tracy:
                profiler_tracy.set_capture_mask(0)
                profiler_tracy.end(0)
                profiler_tracy.shutdown()
        except RuntimeError:
            # Tracy plugin was not loaded, so profiler never started - skip checks.
            pass

        # Disable logging before shutdown to keep the log clean
        # Warnings at this point don't matter as the python process is about to be terminated
        logging = carb.logging.acquire_logging()
        logging.set_level_threshold(carb.logging.LEVEL_ERROR)

        # App shutdown is disabled to prevent crashes on shutdown. Terminating carb is faster
        self._app.shutdown()
        self._framework.unload_all_plugins()
        sys.exit(0)


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
