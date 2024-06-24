# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import torch
from collections.abc import Sequence
from typing import Any

import carb
import omni.usd
from omni.isaac.cloner import GridCloner
from omni.isaac.core.prims import XFormPrimView
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from .interactive_scene_cfg import InteractiveSceneCfg


class InteractiveScene:
    """A scene that contains entities added to the simulation.

    The interactive scene parses the :class:`InteractiveSceneCfg` class to create the scene.
    Based on the specified number of environments, it clones the entities and groups them into different
    categories (e.g., articulations, sensors, etc.).

    Each entity is registered to scene based on its name in the configuration class. For example, if the user
    specifies a robot in the configuration class as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.anymal import ANYMAL_C_CFG

        @configclass
        class MySceneCfg(InteractiveSceneCfg):

            robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    Then the robot can be accessed from the scene as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveScene

        # create 128 environments
        scene = InteractiveScene(cfg=MySceneCfg(num_envs=128))

        # access the robot from the scene
        robot = scene["robot"]
        # access the robot based on its type
        robot = scene.articulations["robot"]

    .. note::
        It is important to note that the scene only performs common operations on the entities. For example,
        resetting the internal buffers, writing the buffers to the simulation and updating the buffers from the
        simulation. The scene does not perform any task specific to the entity. For example, it does not apply
        actions to the robot or compute observations from the robot. These tasks are handled by different
        modules called "managers" in the framework. Please refer to the :mod:`isaaclab.managers` sub-package
        for more details.
    """

    def __init__(self, cfg: InteractiveSceneCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.
        """
        # store inputs
        self.cfg = cfg
        # initialize scene elements
        self._terrain = None
        self._articulations = dict()
        self._rigid_objects = dict()
        self._sensors = dict()
        self._extras = dict()
        # obtain the current stage
        self.stage = omni.usd.get_context().get_stage()
        # physics scene path
        self._physics_scene_path = None
        # prepare cloner for environment replication
        self.cloner = GridCloner(spacing=self.cfg.env_spacing)
        self.cloner.define_base_env(self.env_ns)
        self.env_prim_paths = self.cloner.generate_paths(f"{self.env_ns}/env", self.cfg.num_envs)
        # create source prim
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        # clone the env xform
        env_origins = self.cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=False,
            copy_from_source=True,
        )
        self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)
        self._global_prim_paths = list()
        if self._is_scene_setup_from_cfg():
            # add entities from config
            self._add_entities_from_cfg()
            # replicate physics if we have more than one environment
            # this is done to make scene initialization faster at play time
            if self.cfg.replicate_physics and self.cfg.num_envs > 1:
                self.cloner.replicate_physics(
                    source_prim_path=self.env_prim_paths[0],
                    prim_paths=self.env_prim_paths,
                    base_env_path=self.env_ns,
                    root_path=self.env_regex_ns.replace(".*", ""),
                )
            self.filter_collisions(self._global_prim_paths)

    def clone_environments(self, copy_from_source: bool = False):
        """Creates clones of the environment ``/World/envs/env_0``.

        Args:
            copy_from_source: (bool): If set to False, clones inherit from /World/envs/env_0 and mirror its changes.
            If True, clones are independent copies of the source prim and won't reflect its changes (start-up time
            may increase). Defaults to False.
        """
        self.cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=self.cfg.replicate_physics,
            copy_from_source=copy_from_source,
        )

    def filter_collisions(self, global_prim_paths: list[str] = []):
        """Filter environments collisions.

        Disables collisions between the environments in ``/World/envs/env_.*`` and enables collisions with the prims
        in global prim paths (e.g. ground plane).

        Args:
            global_prim_paths: The global prim paths to enable collisions with.
        """
        # obtain the current physics scene
        physics_scene_prim_path = self.physics_scene_path

        # filter collisions within each environment instance
        self.cloner.filter_collisions(
            physics_scene_prim_path,
            "/World/collisions",
            self.env_prim_paths,
            global_paths=global_prim_paths,
        )

    def __str__(self) -> str:
        """Returns a string representation of the scene."""
        msg = f"<class {self.__class__.__name__}>\n"
        msg += f"\tNumber of environments: {self.cfg.num_envs}\n"
        msg += f"\tEnvironment spacing   : {self.cfg.env_spacing}\n"
        msg += f"\tSource prim name      : {self.env_prim_paths[0]}\n"
        msg += f"\tGlobal prim paths     : {self._global_prim_paths}\n"
        msg += f"\tReplicate physics     : {self.cfg.replicate_physics}"
        return msg

    """
    Properties.
    """

    @property
    def physics_scene_path(self):
        """Search the stage for the physics scene"""
        if self._physics_scene_path is None:
            for prim in self.stage.Traverse():
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    self._physics_scene_path = prim.GetPrimPath()
                    carb.log_info(f"Physics scene prim path: {self._physics_scene_path}")
                    break
        return self._physics_scene_path

    @property
    def physics_dt(self) -> float:
        """The physics timestep of the scene."""
        return sim_utils.SimulationContext.instance().get_physics_dt()  # pyright: ignore [reportOptionalMemberAccess]

    @property
    def device(self) -> str:
        """The device on which the scene is created."""
        return sim_utils.SimulationContext.instance().device  # pyright: ignore [reportOptionalMemberAccess]

    @property
    def env_ns(self) -> str:
        """The namespace ``/World/envs`` in which all environments created.

        The environments are present w.r.t. this namespace under "env_{N}" prim,
        where N is a natural number.
        """
        return "/World/envs"

    @property
    def env_regex_ns(self) -> str:
        """The namespace ``/World/envs/env_.*`` in which all environments created."""
        return f"{self.env_ns}/env_.*"

    @property
    def num_envs(self) -> int:
        """The number of environments handled by the scene."""
        return self.cfg.num_envs

    @property
    def env_origins(self) -> torch.Tensor:
        """The origins of the environments in the scene. Shape is (num_envs, 3)."""
        if self._terrain is not None:
            return self._terrain.env_origins
        else:
            return self._default_env_origins

    @property
    def terrain(self) -> TerrainImporter | None:
        """The terrain in the scene. If None, then the scene has no terrain.

        Note:
            We treat terrain separate from :attr:`extras` since terrains define environment origins and are
            handled differently from other miscellaneous entities.
        """
        return self._terrain

    @property
    def articulations(self) -> dict[str, Articulation]:
        """A dictionary of articulations in the scene."""
        return self._articulations

    @property
    def rigid_objects(self) -> dict[str, RigidObject]:
        """A dictionary of rigid objects in the scene."""
        return self._rigid_objects

    @property
    def sensors(self) -> dict[str, SensorBase]:
        """A dictionary of the sensors in the scene, such as cameras and contact reporters."""
        return self._sensors

    @property
    def extras(self) -> dict[str, XFormPrimView]:
        """A dictionary of miscellaneous simulation objects that neither inherit from assets nor sensors.

        The keys are the names of the miscellaneous objects, and the values are the `XFormPrimView`_
        of the corresponding prims.

        As an example, lights or other props in the scene that do not have any attributes or properties that you
        want to alter at runtime can be added to this dictionary.

        Note:
            These are not reset or updated by the scene. They are mainly other prims that are not necessarily
            handled by the interactive scene, but are useful to be accessed by the user.

        .. _XFormPrimView: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#omni.isaac.core.prims.XFormPrimView

        """
        return self._extras

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the scene entities.

        Args:
            env_ids: The indices of the environments to reset.
                Defaults to None (all instances).
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_ids)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.reset(env_ids)
        # -- flush physics sim view if called in extension mode
        # this is needed when using PhysX GPU pipeline since the data needs to be sent to the underlying
        # PhysX buffers that might live on a separate device
        # note: In standalone mode, this method is called in the `step()` method of the simulation context.
        #   So we only need to flush when running in extension mode.
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            sim_utils.SimulationContext.instance().physics_sim_view.flush()  # pyright: ignore [reportOptionalMemberAccess]

    def write_data_to_sim(self):
        """Writes the data of the scene entities to the simulation."""
        # -- assets
        for articulation in self._articulations.values():
            articulation.write_data_to_sim()
        for rigid_object in self._rigid_objects.values():
            rigid_object.write_data_to_sim()
        # -- flush physics sim view if called in extension mode
        # this is needed when using PhysX GPU pipeline since the data needs to be sent to the underlying
        # PhysX buffers that might live on a separate device
        # note: In standalone mode, this method is called in the `step()` method of the simulation context.
        #   So we only need to flush when running in extension mode.
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            sim_utils.SimulationContext.instance().physics_sim_view.flush()  # pyright: ignore [reportOptionalMemberAccess]

    def update(self, dt: float) -> None:
        """Update the scene entities.

        Args:
            dt: The amount of time passed from last :meth:`update` call.
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.update(dt)
        for rigid_object in self._rigid_objects.values():
            rigid_object.update(dt)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

    """
    Operations: Iteration.
    """

    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        """
        all_keys = ["terrain"]
        for asset_family in [self._articulations, self._rigid_objects, self._sensors, self._extras]:
            all_keys += list(asset_family.keys())
        return all_keys

    def __getitem__(self, key: str) -> Any:
        """Returns the scene entity with the given key.

        Args:
            key: The key of the scene entity.

        Returns:
            The scene entity.
        """
        # check if it is a terrain
        if key == "terrain":
            return self._terrain

        all_keys = ["terrain"]
        # check if it is in other dictionaries
        for asset_family in [self._articulations, self._rigid_objects, self._sensors, self._extras]:
            out = asset_family.get(key)
            # if found, return
            if out is not None:
                return out
            all_keys += list(asset_family.keys())
        # if not found, raise error
        raise KeyError(f"Scene entity with key '{key}' not found. Available Entities: '{all_keys}'")

    """
    Internal methods.
    """

    def _is_scene_setup_from_cfg(self):
        return any(
            not (asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None)
            for asset_name, asset_cfg in self.cfg.__dict__.items()
        )

    def _add_entities_from_cfg(self):
        """Add scene entities from the config."""
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            # resolve regex
            asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            # create asset
            if isinstance(asset_cfg, TerrainImporterCfg):
                # terrains are special entities since they define environment origins
                asset_cfg.num_envs = self.cfg.num_envs
                asset_cfg.env_spacing = self.cfg.env_spacing
                self._terrain = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, SensorBaseCfg):
                # Update target frame path(s)' regex name space for FrameTransformer
                if isinstance(asset_cfg, FrameTransformerCfg):
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr

                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, AssetBaseCfg):
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                self._extras[asset_name] = XFormPrimView(asset_cfg.prim_path, reset_xform_properties=False)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
