# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_physx.assets import DeformableObject, SurfaceGripper

import torch
import warp as wp

from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id
from isaaclab.sim.views import XformPrimView
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

# Note: This is a temporary import for the VisuoTactileSensorCfg class.
# It will be removed once the VisuoTactileSensor class is added to the core Isaac Lab framework.
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from .interactive_scene_cfg import InteractiveSceneCfg

# import logger
logger = logging.getLogger(__name__)


class InteractiveScene:
    """A scene that contains entities added to the simulation.

    The interactive scene parses the :class:`InteractiveSceneCfg` class to create the scene.
    Based on the specified number of environments, it clones the entities and groups them into different
    categories (e.g., articulations, sensors, etc.).

    Cloning can be performed in two ways:

    * For tasks where all environments contain the same assets, a more performant cloning paradigm
      can be used to allow for faster environment creation. This is specified by the ``replicate_physics`` flag.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=True))

    * For tasks that require having separate assets in the environments, ``replicate_physics`` would have to
      be set to False, which will add some costs to the overall startup time.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=False))

    Each entity is registered to scene based on its name in the configuration class. For example, if the user
    specifies a robot in the configuration class as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # ANYmal-C robot spawned in each environment
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

    If the :class:`InteractiveSceneCfg` class does not include asset entities, the cloning process
    can still be triggered if assets were added to the stage outside of the :class:`InteractiveScene` class:

    .. code-block:: python

        scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128, replicate_physics=True))
        scene.clone_environments()

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
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        # initialize scene elements
        self._terrain = None
        self._articulations = dict()
        self._deformable_objects = dict()
        self._rigid_objects = dict()
        self._rigid_object_collections = dict()
        self._sensors = dict()
        self._surface_grippers = dict()
        self._extras = dict()
        # get stage handle
        self.sim = SimulationContext.instance()
        self.stage = get_current_stage()
        self.stage_id = get_current_stage_id()
        # physics backend and clone fn (PhysX uses PhysxSchema/physx_replicate; Newton uses its own worlds)
        self.physics_backend = self.sim.physics_manager.__name__
        if "Physx" in self.physics_backend:
            from isaaclab_physx.cloner import physx_replicate

            physics_clone_fn = physx_replicate
        elif "Newton" in self.physics_backend:
            from isaaclab_newton.cloner import newton_replicate

            physics_clone_fn = newton_replicate
        else:
            raise ValueError(f"Unsupported physics backend: {self.physics_backend}")
        # physics scene path
        self._physics_scene_path = None
        # prepare cloner for environment replication
        self.env_prim_paths = [f"{self.env_ns}/env_{i}" for i in range(self.cfg.num_envs)]

        self.cloner_cfg = cloner.TemplateCloneCfg(
            clone_regex=self.env_regex_ns,
            clone_in_fabric=self.cfg.clone_in_fabric,
            device=self.device,
            physics_clone_fn=physics_clone_fn,
        )

        # create source prim
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        self.stage.DefinePrim(self.cloner_cfg.template_root, "Xform")
        self.env_fmt = self.env_regex_ns.replace(".*", "{}")
        # allocate env indices
        self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        self._default_env_origins, _ = cloner.grid_transforms(self.num_envs, self.cfg.env_spacing, device=self.device)
        # copy empty prim of env_0 to env_1, env_2, ..., env_{num_envs-1} with correct location.
        cloner.usd_replicate(
            self.stage, [self.env_fmt.format(0)], [self.env_fmt], self._ALL_INDICES, positions=self._default_env_origins
        )

        self._global_prim_paths = list()
        if self._is_scene_setup_from_cfg():
            self._add_entities_from_cfg()
            self.clone_environments(copy_from_source=(not self.cfg.replicate_physics))
            # Collision filtering is PhysX-specific (PhysxSchema.PhysxSceneAPI)
            if self.cfg.filter_collisions and "Physx" in self.physics_backend:
                self.filter_collisions(self._global_prim_paths)

    def clone_environments(self, copy_from_source: bool = False):
        """Creates clones of the environment ``/World/envs/env_0``.

        Args:
            copy_from_source: (bool): If set to False, clones inherit from /World/envs/env_0 and mirror its changes.
            If True, clones are independent copies of the source prim and won't reflect its changes (start-up time
            may increase). Defaults to False.
        """
        # PhysX-only: set env id bit count for replicated physics. Newton handles env separation in its own API.
        if self.cfg.replicate_physics and "Physx" in self.physics_backend:
            prim = self.stage.GetPrimAtPath("/physicsScene")
            prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

        if self._is_scene_setup_from_cfg():
            self.cloner_cfg.clone_physics = not copy_from_source
            cloner.clone_from_template(self.stage, num_clones=self.num_envs, template_clone_cfg=self.cloner_cfg)
        else:
            mapping = torch.ones((1, self.num_envs), device=self.device, dtype=torch.bool)
            replicate_args = [self.env_fmt.format(0)], [self.env_fmt], self._ALL_INDICES, mapping

            if not copy_from_source:
                # skip physx cloning, this means physx will walk and parse the stage one by one faithfully
                self.cloner_cfg.physics_clone_fn(self.stage, *replicate_args, device=self.cloner_cfg.device)
            cloner.usd_replicate(self.stage, *replicate_args, positions=self._default_env_origins)

    def filter_collisions(self, global_prim_paths: list[str] | None = None):
        """Filter environments collisions.

        Disables collisions between the environments in ``/World/envs/env_.*`` and enables collisions with the prims
        in global prim paths (e.g. ground plane).

        Args:
            global_prim_paths: A list of global prim paths to enable collisions with.
                Defaults to None, in which case no global prim paths are considered.
        """
        # validate paths in global prim paths
        if global_prim_paths is None:
            global_prim_paths = []
        else:
            # remove duplicates in paths
            global_prim_paths = list(set(global_prim_paths))

        # if "/World/collisions" already exists in the stage, we don't filter again
        if self.stage.GetPrimAtPath("/World/collisions"):
            return

        # set global prim paths list if not previously defined
        if len(self._global_prim_paths) < 1:
            self._global_prim_paths += global_prim_paths

        # filter collisions within each environment instance
        cloner.filter_collisions(
            self.stage,
            self.physics_scene_path,
            "/World/collisions",
            self.env_prim_paths,
            global_paths=self._global_prim_paths,
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
    def physics_scene_path(self) -> str:
        """The path to the USD Physics Scene."""
        if self._physics_scene_path is None:
            for prim in self.stage.Traverse():
                if "PhysxSceneAPI" in prim.GetAppliedSchemas():
                    self._physics_scene_path = prim.GetPrimPath().pathString
                    logger.info(f"Physics scene prim path: {self._physics_scene_path}")
                    break
            if self._physics_scene_path is None:
                raise RuntimeError("No physics scene found! Please make sure one exists.")
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
    def deformable_objects(self) -> dict[str, DeformableObject]:
        """A dictionary of deformable objects in the scene."""
        return self._deformable_objects

    @property
    def rigid_objects(self) -> dict[str, RigidObject]:
        """A dictionary of rigid objects in the scene."""
        return self._rigid_objects

    @property
    def rigid_object_collections(self) -> dict[str, RigidObjectCollection]:
        """A dictionary of rigid object collections in the scene."""
        return self._rigid_object_collections

    @property
    def sensors(self) -> dict[str, SensorBase]:
        """A dictionary of the sensors in the scene, such as cameras and contact reporters."""
        return self._sensors

    @property
    def surface_grippers(self) -> dict[str, SurfaceGripper]:
        """A dictionary of the surface grippers in the scene."""
        return self._surface_grippers

    @property
    def extras(self) -> dict[str, XformPrimView]:
        """A dictionary of miscellaneous simulation objects that neither inherit from assets nor sensors.

        The keys are the names of the miscellaneous objects, and the values are the
        :class:`~isaaclab.sim.views.XformPrimView` instances of the corresponding prims.

        As an example, lights or other props in the scene that do not have any attributes or properties that you
        want to alter at runtime can be added to this dictionary.

        Note:
            These are not reset or updated by the scene. They are mainly other prims that are not necessarily
            handled by the interactive scene, but are useful to be accessed by the user.

        """
        return self._extras

    @property
    def state(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """A dictionary of the state of the scene entities in the simulation world frame.

        Please refer to :meth:`get_state` for the format.
        """
        return self.get_state(is_relative=False)

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
        for deformable_object in self._deformable_objects.values():
            deformable_object.reset(env_ids)
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_ids)
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.reset(env_ids)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.reset(env_ids)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.reset(env_ids)

    def write_data_to_sim(self):
        """Writes the data of the scene entities to the simulation."""
        # -- assets
        for articulation in self._articulations.values():
            articulation.write_data_to_sim()
        for deformable_object in self._deformable_objects.values():
            deformable_object.write_data_to_sim()
        for rigid_object in self._rigid_objects.values():
            rigid_object.write_data_to_sim()
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.write_data_to_sim()
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.write_data_to_sim()

    def update(self, dt: float) -> None:
        """Update the scene entities.

        Args:
            dt: The amount of time passed from last :meth:`update` call.
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.update(dt)
        for deformable_object in self._deformable_objects.values():
            deformable_object.update(dt)
        for rigid_object in self._rigid_objects.values():
            rigid_object.update(dt)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.update(dt)
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.update(dt)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

    """
    Operations: Scene State.
    """

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ):
        """Resets the entities in the scene to the provided state.

        Args:
            state: The state to reset the scene entities to. Please refer to :meth:`get_state` for the format.
            env_ids: The indices of the environments to reset. Defaults to None, in which case
                all environment instances are reset.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # articulations
        for asset_name, articulation in self._articulations.items():
            asset_state = state["articulation"][asset_name]
            # root state
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            # joint state
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            # FIXME: This is not generic as it assumes PD control over the joints.
            #   This assumption does not hold for effort controlled joints.
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)
        # deformable objects
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone()
            if is_relative:
                nodal_position[:, :3] += self.env_origins[env_ids]
            nodal_velocity = asset_state["nodal_velocity"].clone()
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)
        # rigid objects
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        # surface grippers
        for asset_name, surface_gripper in self._surface_grippers.items():
            asset_state = state["gripper"][asset_name]
            surface_gripper.set_grippers_command(asset_state)

        # write data to simulation to make sure initial state is set
        # this propagates the joint targets to the simulation
        self.write_data_to_sim()

    def get_state(self, is_relative: bool = False) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Returns the state of the scene entities.

        Based on the type of the entity, the state comprises of different components.

        * For an articulation, the state comprises of the root pose, root velocity, and joint position and velocity.
        * For a deformable object, the state comprises of the nodal position and velocity.
        * For a rigid object, the state comprises of the root pose and root velocity.

        The returned state is a dictionary with the following format:

        .. code-block:: python

            {
                "articulation": {
                    "entity_1_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                    "entity_2_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                },
                "deformable_object": {
                    "entity_3_name": {
                        "nodal_position": torch.Tensor,
                        "nodal_velocity": torch.Tensor,
                    }
                },
                "rigid_object": {
                    "entity_4_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                    }
                },
            }

        where ``entity_N_name`` is the name of the entity registered in the scene.

        Args:
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.

        Returns:
            A dictionary of the state of the scene entities.
        """
        state = dict()
        # articulations
        state["articulation"] = dict()
        for asset_name, articulation in self._articulations.items():
            asset_state = dict()
            asset_state["root_pose"] = wp.to_torch(articulation.data.root_pose_w).clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = wp.to_torch(articulation.data.root_vel_w).clone()
            asset_state["joint_position"] = wp.to_torch(articulation.data.joint_pos).clone()
            asset_state["joint_velocity"] = wp.to_torch(articulation.data.joint_vel).clone()
            state["articulation"][asset_name] = asset_state
        # deformable objects
        state["deformable_object"] = dict()
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = dict()
            asset_state["nodal_position"] = wp.to_torch(deformable_object.data.nodal_pos_w).clone()
            if is_relative:
                asset_state["nodal_position"][:, :3] -= self.env_origins
            asset_state["nodal_velocity"] = wp.to_torch(deformable_object.data.nodal_vel_w).clone()
            state["deformable_object"][asset_name] = asset_state
        # rigid objects
        state["rigid_object"] = dict()
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = wp.to_torch(rigid_object.data.root_pose_w).clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = wp.to_torch(rigid_object.data.root_vel_w).clone()
            state["rigid_object"][asset_name] = asset_state
        # surface grippers
        state["gripper"] = dict()
        for asset_name, gripper in self._surface_grippers.items():
            state["gripper"][asset_name] = wp.to_torch(gripper.state).clone()
        return state

    """
    Operations: Iteration.
    """

    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        """
        all_keys = ["terrain"]
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._sensors,
            self._surface_grippers,
            self._extras,
        ]:
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
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._sensors,
            self._surface_grippers,
            self._extras,
        ]:
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

    def _is_scene_setup_from_cfg(self) -> bool:
        """Check if scene entities are setup from the config or not.

        Returns:
            True if scene entities are setup from the config, False otherwise.
        """
        return any(
            not (asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None)
            for asset_name, asset_cfg in self.cfg.__dict__.items()
        )

    def _add_entities_from_cfg(self):  # noqa: C901
        """Add scene entities from the config."""
        from isaaclab_physx.assets import DeformableObjectCfg, SurfaceGripperCfg  # noqa: PLC0415

        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # Process non-sensor entities before sensors so that asset prims exist in the template
        # when sensors (e.g. cameras attached to robot links) need to spawn under them.
        all_items = [
            (k, v)
            for k, v in self.cfg.__dict__.items()
            if k not in InteractiveSceneCfg.__dataclass_fields__ and v is not None
        ]
        ordered_items = [(k, v) for k, v in all_items if not isinstance(v, SensorBaseCfg)] + [
            (k, v) for k, v in all_items if isinstance(v, SensorBaseCfg)
        ]

        for asset_name, asset_cfg in ordered_items:
            # resolve prim_path with env regex
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            # set spawn_path on spawner if cloning is needed
            if hasattr(asset_cfg, "spawn") and asset_cfg.spawn is not None:
                if hasattr(asset_cfg, "prim_path") and self.env_ns in asset_cfg.prim_path:
                    template_base = asset_cfg.prim_path.replace(self.env_regex_ns, self.cloner_cfg.template_root)
                    proto_id = self.cloner_cfg.template_prototype_identifier
                    if isinstance(asset_cfg, SensorBaseCfg):
                        # Sensor may be nested under a proto_asset_N prim (e.g. a camera on a robot
                        # link). Search for the actual template location so spawning succeeds even
                        # though the parent asset lives at template_root/<Asset>/proto_asset_0/...
                        asset_cfg.spawn.spawn_path = self._resolve_sensor_template_spawn_path(template_base, proto_id)
                    else:
                        asset_cfg.spawn.spawn_path = f"{template_base}/{proto_id}_.*"
                else:
                    # No cloning - spawn directly at prim_path
                    asset_cfg.spawn.spawn_path = asset_cfg.prim_path
            # create asset
            if isinstance(asset_cfg, TerrainImporterCfg):
                # terrains are special entities since they define environment origins
                asset_cfg.num_envs = self.cfg.num_envs
                asset_cfg.env_spacing = self.cfg.env_spacing
                self._terrain = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, DeformableObjectCfg):
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                    # set spawn_path on spawner if cloning is needed
                    if hasattr(rigid_object_cfg, "spawn") and rigid_object_cfg.spawn is not None:
                        if self.env_ns in rigid_object_cfg.prim_path:
                            spawn_tmpl = rigid_object_cfg.prim_path.replace(
                                self.env_regex_ns, self.cloner_cfg.template_root
                            )
                            proto_id = self.cloner_cfg.template_prototype_identifier
                            rigid_object_cfg.spawn.spawn_path = f"{spawn_tmpl}/{proto_id}_.*"
                        else:
                            rigid_object_cfg.spawn.spawn_path = rigid_object_cfg.prim_path
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths
            elif isinstance(asset_cfg, SurfaceGripperCfg):
                # add surface grippers to scene
                self._surface_grippers[asset_name] = asset_cfg.class_type(asset_cfg)
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
                elif isinstance(asset_cfg, VisuoTactileSensorCfg):
                    if hasattr(asset_cfg, "camera_cfg") and asset_cfg.camera_cfg is not None:
                        asset_cfg.camera_cfg.prim_path = asset_cfg.camera_cfg.prim_path.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )
                    if (
                        hasattr(asset_cfg, "contact_object_prim_path_expr")
                        and asset_cfg.contact_object_prim_path_expr is not None
                    ):
                        asset_cfg.contact_object_prim_path_expr = asset_cfg.contact_object_prim_path_expr.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )

                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, AssetBaseCfg):
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.spawn.spawn_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                self._extras[asset_name] = XformPrimView(asset_cfg.prim_path, device=self.device, stage=self.stage)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")

            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths

    def _resolve_sensor_template_spawn_path(self, template_base: str, proto_id: str) -> str:
        """Resolve the actual template spawn path for a sensor nested under a proto_asset prim.

        Sensors parented to robot links live inside ``proto_asset_0`` rather than directly under
        the template root.  For example, a wrist camera at
        ``/World/template/Robot/panda_hand/wrist_cam`` is actually spawned at
        ``/World/template/Robot/proto_asset_0/panda_hand/wrist_cam``.

        This method inserts a ``proto_id_.*`` wildcard one level below the template root and
        searches for the concrete parent prim so the camera spawner can find it.

        Args:
            template_base: Template path derived by replacing the env regex with the template root.
                Example: ``/World/template/Robot/panda_hand/wrist_cam``.
            proto_id: Prototype identifier prefix (e.g. ``proto_asset``).

        Returns:
            Concrete spawn path (e.g. ``/World/template/Robot/proto_asset_0/panda_hand/wrist_cam``)
            if the parent is found, otherwise ``template_base/proto_id_.*`` as a fallback.
        """
        template_root = self.cloner_cfg.template_root
        # rel = e.g. "Robot/panda_hand/wrist_cam"
        rel = template_base[len(template_root) + 1 :]
        # asset = "Robot", remainder = "panda_hand/wrist_cam"
        asset, _, remainder = rel.partition("/")
        if not remainder:
            return f"{template_base}/{proto_id}_.*"

        # parent = "panda_hand", leaf = "wrist_cam"
        parent, _, leaf = remainder.rpartition("/")
        search = (
            f"{template_root}/{asset}/{proto_id}_.*/{parent}" if parent else f"{template_root}/{asset}/{proto_id}_.*"
        )
        found = sim_utils.find_matching_prim_paths(search)
        return f"{found[0]}/{leaf}" if found else f"{template_base}/{proto_id}_.*"
