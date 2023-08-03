# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx
from omni.isaac.core.simulation_context import SimulationContext as _SimulationContext
from pxr import PhysxSchema

from .simulation_cfg import SimulationCfg


class SimulationContext(_SimulationContext):
    """A class to control simulation-related events such as physics stepping and rendering.

    It wraps the ``SimulationContext`` class from ``omni.isaac.core`` and adds some additional
    functionality such as setting up the simulation context with a configuration object.
    """

    def __init__(self, cfg: SimulationCfg = None):
        """Creates a simulation context to control the simulator.

        Args:
            cfg (SimulationCfg, optional): The configuration of the simulation. Defaults to None,
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
        # set up flatcache interface (default is None)
        # note: need to do this here because super().__init__ calls render and this variable is needed
        self._flatcache_iface = None

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
        # modify the physics material
        material_path = f"{self.cfg.physics_prim_path}/defaultMaterial"
        material_prim = prim_utils.get_prim_at_path(material_path)
        # Apply PhysX Rigid Material schema
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        # Set patch friction property
        improve_patch_friction = self.cfg.default_physics_material.improve_patch_friction
        physx_material_api.CreateImprovePatchFrictionAttr().Set(improve_patch_friction)
        # Set combination mode for coefficients
        combine_mode = self.cfg.default_physics_material.combine_mode
        physx_material_api.CreateFrictionCombineModeAttr().Set(combine_mode)
        physx_material_api.CreateRestitutionCombineModeAttr().Set(combine_mode)

        # check if flatcache is enabled
        # this is needed to flush the flatcache data into Hydra manually when calling `render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        if self.cfg.use_flatcache:
            from omni.physxflatcache import get_physx_flatcache_interface

            # acquire flatcache interface
            self._flatcache_iface = get_physx_flatcache_interface()

    """
    Operations - Override.
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
        if self._flatcache_iface is not None and flush:
            self._flatcache_iface.update(0.0, 0.0)
        # render the simulation
        super().render()

    """
    Operations - New.
    """

    def is_headless(self) -> bool:
        """Returns whether the simulation is running in headless mode.

        Note:
            Headless mode is enabled when the simulator is running without a GUI.
        """
        return self._is_headless
