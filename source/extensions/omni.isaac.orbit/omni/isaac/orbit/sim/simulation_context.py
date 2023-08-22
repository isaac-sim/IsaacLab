# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from typing import Any

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.physx
from omni.isaac.core.simulation_context import SimulationContext as _SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.version import get_version
from pxr import Gf, Usd

from .simulation_cfg import SimulationCfg
from .utils import bind_physics_material


class SimulationContext(_SimulationContext):
    """A class to control simulation-related events such as physics stepping and rendering.

    The simulation context helps control various simulation aspects. This includes:

    * configure the simulator with different settings such as the physics timestep, the number of physics substeps,
      and the physics solver parameters (for more information, see :class:`omni.isaac.orbit.sim.SimulationCfg`)
    * playing, pausing, stepping and stopping the simulation
    * adding and removing callbacks to different simulation events such as physics stepping, rendering, etc.

    This class inherits from the `omni.isaac.core.simulation_context.SimulationContext`_ class and
    adds additional functionalities such as setting up the simulation context with a configuration object,
    exposing other commonly used simulator-related functions, and performing version checks of Isaac Sim
    to ensure compatibility between releases.

    The simulation context is a singleton object. This means that there can only be one instance
    of the simulation context at any given time. This is enforced by the parent class. Therefore, it is
    not possible to create multiple instances of the simulation context. Instead, the simulation context
    can be accessed using the ``instance()`` method.

    .. attention::
        Since we only support the ``torch <https://pytorch.org/>``_ backend for simulation, the
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

    .. _omni.isaac.core.simulation_context.SimulationContext: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#module-omni.isaac.core.simulation_context
    """

    def __init__(self, cfg: SimulationCfg | None = None):
        """Creates a simulation context to control the simulator.

        Args:
            cfg (SimulationCfg | None, optional): The configuration of the simulation. Defaults to None,
                in which case the default configuration is used.
        """
        # store input
        if cfg is None:
            cfg = SimulationCfg()
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
        # disable contact processing in omni.physx if requested
        # note: helpful when creating contact reporting over limited number of objects in the scene
        if self.cfg.disable_contact_processing:
            carb_settings_iface.set_bool("/physics/disableContactProcessing", True)

        # read flag for whether headless mode is enabled
        # note: we read this once since it is not expected to change during runtime
        self._is_headless = not carb_settings_iface.get("/app/window/enabled")
        # enable scene querying if rendering is enabled
        # this is needed for some GUI features
        if not self._is_headless:
            self.cfg.enable_scene_query_support = True
        # set up flatcache/fabric interface (default is None)
        # this is needed to flush the flatcache data into Hydra manually when calling `render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        # note: need to do this here because super().__init__ calls render and this variable is needed
        self._fabric_iface = None
        # read isaac sim version (this includes build tag, release tag etc.)
        # note: we do it once here because it reads the VERSION file from disk and is not expected to change.
        self._isaacsim_version = get_version()

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
            rendering_dt=self.cfg.dt * self.cfg.substeps,
            backend="torch",
            sim_params=sim_params,
            physics_prim_path=self.cfg.physics_prim_path,
            device=self.cfg.device,
        )

    """
    Operations - New.
    """

    def is_headless(self) -> bool:
        """Returns whether the simulation is running in headless mode.

        Note:
            Headless mode is enabled when the simulator is running without a GUI.
        """
        return self._is_headless

    def get_version(self) -> tuple[int, int, int]:
        """Returns the version of the simulator.

        This is a wrapper around the ``omni.isaac.version.get_version()`` function.

        The returned tuple contains the following information:

        * Major version (int): This is the year of the release (e.g. 2022).
        * Minor version (int): This is the half-year of the release (e.g. 1 or 2).
        * Patch version (int): This is the patch number of the release (e.g. 0).
        """
        return int(self._isaacsim_version[2]), int(self._isaacsim_version[3]), int(self._isaacsim_version[4])

    """
    Operations - New utilities.
    """

    @staticmethod
    def set_camera_view(
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        """Set the location and target of the viewport camera in the stage.

        Note:
            This is a wrapper around the :math:`omni.isaac.core.utils.viewports.set_camera_view` function.
            It is provided here for convenience to reduce the amount of imports needed.

        Args:
            eye (tuple[float, float, float]): The location of the camera eye.
            target (tuple[float, float, float]): The location of the camera target.
            camera_prim_path (str, optional): The path to the camera primitive in the stage. Defaults to
                "/OmniverseKit_Persp".
        """
        set_camera_view(eye, target, camera_prim_path)

    def set_setting(self, name: str, value: Any):
        """Set simulation settings using the Carbonite SDK.

        .. note::
            If the input setting name does not exist, it will be created. If it does exist, the value will be
            overwritten. Please make sure to use the correct setting name.

            To understand the settings interface, please refer to the
            `Carbonite SDK <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/settings.html>`_
            documentation.

        Args:
            name (str): The name of the setting.
            value (Any): The value of the setting.
        """
        self._settings.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Read the simulation setting using the Carbonite SDK.

        Args:
            name (str): The name of the setting.

        Returns:
            Any: The value of the setting.
        """
        return self._settings.get(name)

    """
    Operations - Override (standalone)
    """

    def reset(self, soft: bool = False):
        # need to load all "physics" information from the USD file
        # FIXME: Remove this for Isaac Sim 2023.1 release if it will be fixed in the core.
        if not soft:
            omni.physx.acquire_physx_interface().force_load_physics_from_usd()
        # play the simulation
        super().reset(soft=soft)

    def step(self, render: bool = True):
        # override render settings if we are in headless mode
        if self._is_headless:
            render = False
        # step the simulation
        super().step(render=render)

    def render(self, flush: bool = True):
        """Refreshes the rendering components including UI elements and view-ports.

        Args:
            flush (bool, optional): Whether to flush the flatcache data to update Hydra textures.
        """
        # manually flush the flatcache data to update Hydra textures
        if self._fabric_iface is not None and flush:
            self._fabric_iface.update(0.0, 0.0)
        # render the simulation
        super().render()

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
    Initialization - Override.
    """

    def _init_stage(self, *args, **kwargs) -> Usd.Stage:
        _ = super()._init_stage(*args, **kwargs)
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

    """
    Helper Functions
    """

    def _set_additional_physx_params(self):
        """Sets additional PhysX parameters that are not directly supported by the parent class."""
        # obtain the physics scene api
        physics_scene = self._physics_context._physics_scene  # pyright: ignore [reportPrivateUsage]
        physx_scene_api = self._physics_context._physx_scene_api  # pyright: ignore [reportPrivateUsage]
        # assert that scene api is not None
        if physx_scene_api is None:
            raise RuntimeError("Physics scene API is None! Please create the scene first.")
        # set parameters not directly supported by the constructor
        # -- Continuous Collision Detection (CCD)
        # ref: https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/AdvancedCollisionDetection.html?highlight=ccd#continuous-collision-detection
        self._physics_context.enable_ccd(self.cfg.physx.enable_ccd)
        # -- GPU collision stack size
        physx_scene_api.CreateGpuCollisionStackSizeAttr(self.cfg.physx.gpu_collision_stack_size)

        # -- Gravity
        # note: Isaac sim only takes the "up-axis" as the gravity direction. But physics allows any direction so we
        #  need to convert the gravity vector to a direction and magnitude pair explicitly.
        gravity = np.asarray(self.cfg.gravity)
        gravity_magnitude = np.linalg.norm(gravity)
        gravity_direction = gravity / gravity_magnitude
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
        physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)

        # simulation iteration count
        if self.get_version()[0] == 2022:
            # position and velocity iteration counts
            physx_scene_api.CreateMinIterationCountAttr(self.cfg.physx.min_position_iteration_count)
            physx_scene_api.CreateMaxIterationCountAttr(self.cfg.physx.max_position_iteration_count)
        else:
            # position iteration count
            physx_scene_api.CreateMinPositionIterationCountsAttr(self.cfg.physx.min_position_iteration_count)
            physx_scene_api.CreateMaxPositionIterationCountsAttr(self.cfg.physx.max_position_iteration_count)
            # velocity iteration count
            physx_scene_api.CreateMinVelocityIterationCountsAttr(self.cfg.physx.min_velocity_iteration_count)
            physx_scene_api.CreateMaxVelocityIterationCountsAttr(self.cfg.physx.max_velocity_iteration_count)

        # create the default physics material
        # this material is used when no material is specified for a primitive
        # check: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-materials
        material_path = f"{self.cfg.physics_prim_path}/defaultMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        # bind the physics material to the scene
        bind_physics_material(self.cfg.physics_prim_path, material_path)

    def _load_fabric_interface(self):
        """Loads the flatcache/fabric interface if enabled."""
        # check isaac sim version
        # note: flatcache is called fabric in isaac sim 2023.x
        #   in isaac sim 2022.x, we use physx-flatcache module
        #   in isaac sim 2023.x, we use physx-fabric module
        if self.get_version()[0] == 2022:
            # check if flatcache is enabled
            if self.cfg.use_flatcache:
                from omni.physxflatcache import get_physx_flatcache_interface

                # acquire flatcache interface
                self._fabric_iface = get_physx_flatcache_interface()
        else:
            # check if fabric is enabled
            if self.cfg.use_fabric:
                from omni.physxfabric import get_physx_fabric_interface

                # acquire fabric interface
                self._fabric_iface = get_physx_fabric_interface()
