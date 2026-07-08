# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_physx.assets import SurfaceGripper

    from isaaclab.renderers.base_renderer import BaseRenderer

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.physics.scene_data_requirements import aggregate_requirements, resolve_scene_data_requirements
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id
from isaaclab.sim.views import FrameView
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

# Note: This is a temporary import for the VisuoTactileSensorCfg class.
# It will be removed once the VisuoTactileSensor class is added to the core Isaac Lab framework.
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from .interactive_scene_cfg import InteractiveSceneCfg

if TYPE_CHECKING:
    from pxr import Sdf  # noqa: F401

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
        from isaaclab.utils.configclass import configclass

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
    can still be triggered by constructing assets directly on the stage and then calling
    :func:`isaaclab.cloner.replicate` with a single-source :class:`~isaaclab.cloner.ClonePlan`:

    .. code-block:: python

        from isaaclab import cloner
        from isaaclab.assets import Articulation

        scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128, replicate_physics=True))
        robot = Articulation(robot_cfg)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(scene.num_envs, scene.cfg.env_spacing, device=scene.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, scene.num_envs, scene.device, pos)
        cloner.replicate(plan, stage=scene.stage)

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
        self.physics_backend = self.sim.physics_manager.__name__.lower()
        requested_viz_types = set(self.sim.resolve_visualizer_types())
        # physics scene path
        self._physics_scene_path = None
        # prepare cloner for environment replication
        self.cloner_cfg = cloner.CloneCfg(device=self.device)
        env_root = self.cloner_cfg.clone_regex.rsplit("/", 1)[0]
        self.env_prim_paths = [f"{env_root}/env_{i}" for i in range(self.cfg.num_envs)]

        # create source prim
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        # allocate env indices
        self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        # clone env_0 xform to env_1..env_{N-1} at grid origins
        env_origins, _ = cloner.grid_transforms(self.num_envs, self.cfg.env_spacing, device=self.device)
        with cloner.disabled_fabric_change_notifies(self.stage, restore=False):
            cloner.usd_replicate(
                self.stage,
                ["/World/envs/env_0"],
                ["/World/envs/env_{}"],
                self._ALL_INDICES,
                positions=env_origins,
            )

        # Always enter so a ClonePlan is published even when the scene cfg has no entities.
        self._global_prim_paths = list()
        with cloner.ReplicateSession(
            self._collect_asset_cfgs(),
            num_clones=self.num_envs,
            env_spacing=self.cfg.env_spacing,
            device=self.device,
            stage=self.stage,
            clone_strategy=self.cloner_cfg.clone_strategy,
        ):
            if self._is_scene_setup_from_cfg():
                self._add_entities_from_cfg()

        self._aggregate_scene_data_requirements(requested_viz_types)

        # Collision filtering is PhysX-only (matches both physx and ovphysx).
        if self.cfg.filter_collisions and "physx" in self.physics_backend and self._is_scene_setup_from_cfg():
            self.filter_collisions(self._global_prim_paths)

    def _collect_asset_cfgs(self) -> list[Any]:
        """Flatten user-declared cfgs for :func:`~isaaclab.cloner.make_clone_plan`.

        Expands :class:`~isaaclab.assets.RigidObjectCollectionCfg` into its members,
        resolves ``{ENV_REGEX_NS}`` macros, and orders sensors after non-sensors.
        """
        cfg_fields = InteractiveSceneCfg.__dataclass_fields__
        items = [(k, v) for k, v in self.cfg.__dict__.items() if k not in cfg_fields and v is not None]
        ordered_items = [v for _, v in items if not isinstance(v, SensorBaseCfg)]
        ordered_items += [v for _, v in items if isinstance(v, SensorBaseCfg)]

        cfgs: list[Any] = []
        for asset_cfg in ordered_items:
            children = (
                asset_cfg.rigid_objects.values() if isinstance(asset_cfg, RigidObjectCollectionCfg) else [asset_cfg]
            )
            for child in children:
                if hasattr(child, "prim_path"):
                    child.prim_path = child.prim_path.format(ENV_REGEX_NS=self.cloner_cfg.clone_regex)
                cfgs.append(child)
        return cfgs

    def _aggregate_scene_data_requirements(self, visualizer_types=()) -> None:
        """Aggregate scene-data requirements from visualizers and sensor renderers.

        Runs once after :meth:`_add_entities_from_cfg` so all sensors are constructed and
        their renderer types are visible. Pushes the merged :class:`SceneDataRequirement` to
        :class:`SimulationContext` for later consumption by the scene data provider.
        """
        discovered_req = resolve_scene_data_requirements(
            visualizer_types=visualizer_types,
            renderer_types=self._sensor_renderer_types(),
        )
        current_req = self.sim.get_scene_data_requirements()
        requirements = aggregate_requirements((current_req, discovered_req))
        if requirements != current_req:
            self.sim.update_scene_data_requirements(requirements)

    def _sensor_renderer_types(self) -> list[str]:
        """Return renderer type names used by scene sensors (skipping any without a renderer cfg)."""
        return [
            getattr(rcfg, "renderer_type", "default")
            for s in self._sensors.values()
            if (rcfg := getattr(getattr(s, "cfg", None), "renderer_cfg", None)) is not None
        ]

    def initialize_renderers(self) -> list[BaseRenderer]:
        """Pre-create renderer backends for all scene sensors with a ``renderer_cfg``.

        Walks the constructed sensors and registers each unique
        :class:`~isaaclab.renderers.renderer_cfg.RendererCfg` with the
        simulation-scoped :class:`~isaaclab.renderers.render_context.RenderContext`.
        Configs that compare equal share a single backend (see
        :meth:`~isaaclab.renderers.render_context.RenderContext.get_renderer`), so
        calling this method is idempotent and safe to invoke before
        :meth:`~isaaclab.sim.SimulationContext.reset`.

        Pre-creating backends here makes the order of renderer construction
        deterministic (matches sensor registration order) and front-loads logging
        instead of trickling out during the first :meth:`Camera._initialize_impl`.
        :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage` is
        intentionally not invoked here; it runs on first camera initialization
        with the correct ``num_envs`` and final stage.

        Returns:
            The list of unique renderer backends now registered on the
            shared :class:`~isaaclab.renderers.render_context.RenderContext`,
            in sensor registration order.
        """
        ctx = self.sim.render_context
        backends: list[BaseRenderer] = []
        seen: set[int] = set()
        for sensor in self._sensors.values():
            rcfg = getattr(getattr(sensor, "cfg", None), "renderer_cfg", None)
            if rcfg is None:
                continue
            backend = ctx.get_renderer(rcfg)
            if id(backend) not in seen:
                seen.add(id(backend))
                backends.append(backend)
        return backends

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
            # Prefer a prim with PhysxSceneAPI applied (Isaac Sim flow).  Fall
            # back to any UsdPhysics.Scene prim (kitless OvPhysX flow does not
            # load the omni.physx schema, so the auto-created scene only
            # carries the stock USD type without PhysxSceneAPI).
            fallback_path: str | None = None
            for prim in self.stage.Traverse():
                if "PhysxSceneAPI" in prim.GetAppliedSchemas():
                    self._physics_scene_path = prim.GetPrimPath().pathString
                    logger.info(f"Physics scene prim path: {self._physics_scene_path}")
                    break
                if fallback_path is None and prim.GetTypeName() == "PhysicsScene":
                    fallback_path = prim.GetPrimPath().pathString
            if self._physics_scene_path is None and fallback_path is not None:
                self._physics_scene_path = fallback_path
                logger.info(f"Physics scene prim path (no PhysxSceneAPI): {self._physics_scene_path}")
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
    def num_envs(self) -> int:
        """The number of environments handled by the scene."""
        return self.cfg.num_envs

    @property
    def env_origins(self) -> torch.Tensor:
        """Per-env world origins, shape ``(num_envs, 3)``. From the terrain when registered,
        else from the published :class:`~isaaclab.cloner.ClonePlan`.
        """
        if self._terrain is not None:
            return self._terrain.env_origins
        return self.sim.get_clone_plan().positions

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
    def clone_plan(self) -> cloner.ClonePlan | None:
        """Clone plan produced by the most recent replication.

        Forwards to :meth:`SimulationContext.get_clone_plan`, which is the canonical owner.
        The plan records the source paths, destination templates, and the per-env source
        assignment mask. ``None`` until :func:`isaaclab.cloner.replicate` has run.
        """
        return self.sim.get_clone_plan()

    @property
    def extras(self) -> dict[str, FrameView]:
        """A dictionary of miscellaneous simulation objects that neither inherit from assets nor sensors.

        The keys are the names of the miscellaneous objects, and the values are the
        :class:`~isaaclab.sim.views.FrameView` instances of the corresponding prims.

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
        # Scene-wide renderer transform sync once per step when all sensors update,
        # so per-camera fetches do not own this concern (deduped inside RenderContext).
        if not self.cfg.lazy_sensor_update:
            self.sim.render_context.update_transforms(self.sim.get_physics_step_count())

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
            root_pose = asset_state["root_pose"].clone().to(self.device)
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone().to(self.device)
            articulation.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
            # joint state
            joint_position = asset_state["joint_position"].clone().to(self.device)
            joint_velocity = asset_state["joint_velocity"].clone().to(self.device)
            articulation.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
            articulation.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
            # FIXME: This is not generic as it assumes PD control over the joints.
            #   This assumption does not hold for effort controlled joints.
            articulation.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target_index(target=joint_velocity, env_ids=env_ids)
        # deformable objects
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone().to(self.device)
            if is_relative:
                nodal_position[:, :3] += self.env_origins[env_ids]
            nodal_velocity = asset_state["nodal_velocity"].clone().to(self.device)
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)
        # rigid objects
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone().to(self.device)
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone().to(self.device)
            rigid_object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
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
            asset_state["root_pose"] = articulation.data.root_pose_w.torch.clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = articulation.data.root_vel_w.torch.clone()
            asset_state["joint_position"] = articulation.data.joint_pos.torch.clone()
            asset_state["joint_velocity"] = articulation.data.joint_vel.torch.clone()
            state["articulation"][asset_name] = asset_state
        # deformable objects
        state["deformable_object"] = dict()
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = dict()
            asset_state["nodal_position"] = deformable_object.data.nodal_pos_w.torch.clone()
            if is_relative:
                asset_state["nodal_position"][:, :3] -= self.env_origins
            asset_state["nodal_velocity"] = deformable_object.data.nodal_vel_w.torch.clone()
            state["deformable_object"][asset_name] = asset_state
        # rigid objects
        state["rigid_object"] = dict()
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = rigid_object.data.root_pose_w.torch.clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = rigid_object.data.root_vel_w.torch.clone()
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
        from isaaclab_physx.assets import SurfaceGripperCfg  # noqa: PLC0415

        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # Resolve the env-namespace convention from the cloner cfg once for this pass.
        env_regex_ns = self.cloner_cfg.clone_regex
        env_root = env_regex_ns.rsplit("/", 1)[0]
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
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
            # set spawn_path on spawner if cloning is needed
            if hasattr(asset_cfg, "spawn") and asset_cfg.spawn is not None:
                is_multi_spawner = isinstance(
                    asset_cfg.spawn, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)
                )
                if env_root not in asset_cfg.prim_path:
                    asset_cfg.spawn.spawn_path = asset_cfg.prim_path
                elif is_multi_spawner and not asset_cfg.spawn.spawn_paths:
                    raise RuntimeError(f"Clone planning did not assign spawn_paths for '{asset_cfg.prim_path}'.")
                elif not is_multi_spawner and asset_cfg.spawn.spawn_path is None:
                    raise RuntimeError(f"Clone planning did not assign spawn_path for '{asset_cfg.prim_path}'.")
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
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
                    # set spawn_path on spawner if cloning is needed
                    if hasattr(rigid_object_cfg, "spawn") and rigid_object_cfg.spawn is not None:
                        is_multi_spawner = isinstance(
                            rigid_object_cfg.spawn, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)
                        )
                        if env_root not in rigid_object_cfg.prim_path:
                            rigid_object_cfg.spawn.spawn_path = rigid_object_cfg.prim_path
                        elif is_multi_spawner and not rigid_object_cfg.spawn.spawn_paths:
                            raise RuntimeError(
                                f"Clone planning did not assign spawn_paths for '{rigid_object_cfg.prim_path}'."
                            )
                        elif not is_multi_spawner and rigid_object_cfg.spawn.spawn_path is None:
                            raise RuntimeError(
                                f"Clone planning did not assign spawn_path for '{rigid_object_cfg.prim_path}'."
                            )
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
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    asset_cfg.filter_prim_paths_expr = [
                        p.format(ENV_REGEX_NS=env_regex_ns) for p in asset_cfg.filter_prim_paths_expr
                    ]
                    if hasattr(asset_cfg, "sensor_shape_prim_expr") and asset_cfg.sensor_shape_prim_expr:
                        asset_cfg.sensor_shape_prim_expr = [
                            p.format(ENV_REGEX_NS=env_regex_ns) for p in asset_cfg.sensor_shape_prim_expr
                        ]
                    if hasattr(asset_cfg, "filter_shape_prim_expr") and asset_cfg.filter_shape_prim_expr:
                        asset_cfg.filter_shape_prim_expr = [
                            p.format(ENV_REGEX_NS=env_regex_ns) for p in asset_cfg.filter_shape_prim_expr
                        ]
                elif isinstance(asset_cfg, VisuoTactileSensorCfg):
                    if hasattr(asset_cfg, "camera_cfg") and asset_cfg.camera_cfg is not None:
                        asset_cfg.camera_cfg.prim_path = asset_cfg.camera_cfg.prim_path.format(
                            ENV_REGEX_NS=env_regex_ns
                        )
                    if (
                        hasattr(asset_cfg, "contact_object_prim_path_expr")
                        and asset_cfg.contact_object_prim_path_expr is not None
                    ):
                        asset_cfg.contact_object_prim_path_expr = asset_cfg.contact_object_prim_path_expr.format(
                            ENV_REGEX_NS=env_regex_ns
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
                self._extras[asset_name] = FrameView(asset_cfg.prim_path, device=self.device, stage=self.stage)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")

            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
