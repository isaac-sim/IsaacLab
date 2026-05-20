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

from pxr import Sdf

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
        self.physics_backend = self.sim.physics_manager.__name__.lower()
        requested_viz_types = set(self.sim.resolve_visualizer_types())
        if self.physics_backend.startswith("ovphysx"):
            from isaaclab_ovphysx.cloner import ovphysx_replicate

            physics_clone_fn = ovphysx_replicate
        elif self.physics_backend.startswith("physx"):
            from isaaclab_physx.cloner import physx_replicate

            physics_clone_fn = physx_replicate
        elif self.physics_backend.startswith("newton"):
            from isaaclab_newton.cloner import newton_physics_replicate

            physics_clone_fn = newton_physics_replicate
        else:
            raise ValueError(f"Unsupported physics backend: {self.physics_backend}")
        # physics scene path
        self._physics_scene_path = None
        # prepare cloner for environment replication
        self.env_prim_paths = [f"{self.env_ns}/env_{i}" for i in range(self.cfg.num_envs)]

        self.cloner_cfg = cloner.CloneCfg(
            clone_regex=self.env_regex_ns,
            clone_in_fabric=self.cfg.clone_in_fabric,
            device=self.device,
            physics_clone_fn=physics_clone_fn,
            # USD replication runs for every backend.  PhysX/Newton need per-env
            # USD prims for sensor discovery.  For OVPhysX, the per-env USD
            # subtrees are layered on TOP of the physics-side ``physx.clone()``
            # replicas -- PhysX is indifferent to additional USD content and
            # the two layers don't conflict.  Probing whether this assumption
            # holds in practice; revert to ``not startswith("ovphysx")`` if
            # ``physx.clone()`` errors on already-populated targets.
            clone_usd=True,
        )

        # create source prim
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        self.env_fmt = self.env_regex_ns.replace(".*", "{}")
        # allocate env indices
        self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        self._default_env_origins, _ = cloner.grid_transforms(self.num_envs, self.cfg.env_spacing, device=self.device)
        # copy empty prim of env_0 to env_1, env_2, ..., env_{num_envs-1} with correct location.
        # Suspend Fabric's USD notice listener: scene-init is followed by ``SimulationContext.reset``,
        # which does the Fabric resync naturally — re-enabling here would just trigger a redundant batch.
        # Note: ``restore=False`` means the listener stays disabled past this ``with`` block — through
        # ``_add_entities_from_cfg`` and ``clone_environments`` below — until ``SimulationContext.reset``
        # re-enables it. The nested suspension inside ``clone_environments`` becomes a no-op as a result.
        with cloner.disabled_fabric_change_notifies(self.stage, restore=False):
            cloner.usd_replicate(
                self.stage,
                [self.env_fmt.format(0)],
                [self.env_fmt],
                self._ALL_INDICES,
                positions=self._default_env_origins,
            )

        self._global_prim_paths = list()
        has_scene_cfg_entities = self._is_scene_setup_from_cfg()
        if has_scene_cfg_entities:
            self._clone_plan = self._build_clone_plan_from_cfg()
            self._add_entities_from_cfg()
        else:
            self._clone_plan = cloner.ClonePlan(
                sources=(self.env_fmt.format(0),),
                destinations=(self.env_fmt,),
                clone_mask=torch.ones((1, self.num_envs), device=self.device, dtype=torch.bool),
            )

        # Aggregate scene-data requirements from declared visualizers and constructed sensors,
        # then publish to ``SimulationContext`` so downstream providers (constructed later by
        # :meth:`SimulationContext.initialize_visualizers`) see the full picture in one read.
        self._aggregate_scene_data_requirements(requested_viz_types)

        if has_scene_cfg_entities:
            self.clone_environments(copy_from_source=(not self.cfg.replicate_physics))
            # Collision filtering is PhysX-specific (PhysxSchema.PhysxSceneAPI)
            # Intentionally matches both physx and ovphysx (both are PhysX-based)
            if self.cfg.filter_collisions and "physx" in self.physics_backend:
                self.filter_collisions(self._global_prim_paths)

    def _build_clone_plan_from_cfg(self) -> cloner.ClonePlan | None:
        """Build a clone plan from scene cfg spawn variants and write planned spawn paths.

        Returns ``None`` when the cfg has no env-scoped spawned assets.
        """

        def num_variants(spawn_cfg) -> int:
            if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
                return len(spawn_cfg.assets_cfg)
            if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
                return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
            return 1

        def set_spawn_paths(spawn_cfg, paths: list[str | None]) -> None:
            if isinstance(spawn_cfg, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)):
                spawn_cfg.spawn_paths = paths
            else:
                active = [path for path in paths if path is not None]
                if len(active) != 1:
                    raise ValueError("Single spawner expects exactly one planned source path.")
                spawn_cfg.spawn_path = active[0]

        cfg_fields = InteractiveSceneCfg.__dataclass_fields__
        items = [(k, v) for k, v in self.cfg.__dict__.items() if k not in cfg_fields and v is not None]
        ordered_items = [item for item in items if not isinstance(item[1], SensorBaseCfg)]
        ordered_items += [item for item in items if isinstance(item[1], SensorBaseCfg)]

        # One group is one prim path template plus its spawn variants.
        groups = []
        for _, asset_cfg in ordered_items:
            cfgs = asset_cfg.rigid_objects.values() if isinstance(asset_cfg, RigidObjectCollectionCfg) else [asset_cfg]
            for cfg in (cfg for cfg in cfgs if hasattr(cfg, "prim_path")):
                prim_path = cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                if not hasattr(cfg, "spawn") or cfg.spawn is None or self.env_ns not in prim_path:
                    continue
                if (count := num_variants(cfg.spawn)) <= 0:
                    raise ValueError(f"Spawner at '{prim_path}' must have at least one variant.")
                groups.append((cfg.spawn, prim_path.replace(self.env_regex_ns, self.env_fmt), count))

        if not groups:
            return None

        # Homogeneous scenes still spawn sources at env_0, but publish the simpler env-root plan.
        if all(count == 1 for _, _, count in groups):
            for spawn_cfg, destination, _ in groups:
                set_spawn_paths(spawn_cfg, [destination.format(0)])
            return cloner.ClonePlan(
                sources=(self.env_fmt.format(0),),
                destinations=(self.env_fmt,),
                clone_mask=torch.ones((1, self.num_envs), device=self.device, dtype=torch.bool),
            )

        plan = cloner.make_clone_plan(
            [[destination.format(i) for i in range(count)] for _, destination, count in groups],
            [destination for _, destination, _ in groups],
            self.num_envs,
            self.cloner_cfg.clone_strategy,
            self.device,
        )

        # Move each planned source row to the first environment that actually uses it.
        row = 0
        sources = list(plan.sources)
        for spawn_cfg, destination, count in groups:
            mask = plan.clone_mask[row : row + count]
            env_ids = mask.to(torch.int).argmax(dim=1).tolist()
            active = mask.any(dim=1).tolist()
            paths = [destination.format(env_id) if is_active else None for env_id, is_active in zip(env_ids, active)]
            for i, path in zip(range(row, row + count), paths):
                if path is not None:
                    sources[i] = path
            set_spawn_paths(spawn_cfg, paths)
            row += count

        plan = cloner.ClonePlan(sources=tuple(sources), destinations=plan.destinations, clone_mask=plan.clone_mask)
        logger.debug("Built heterogeneous ClonePlan with %d source rows.", len(plan.sources))
        return plan

    def clone_environments(self, copy_from_source: bool = False):
        """Creates clones of the environment ``/World/envs/env_0``.

        Args:
            copy_from_source: (bool): If set to False, clones inherit from /World/envs/env_0 and mirror its changes.
            If True, clones are independent copies of the source prim and won't reflect its changes (start-up time
            may increase). Defaults to False.
        """
        plan = self._clone_plan
        assert self.sim is not None
        if plan is None:
            self.sim.set_clone_plan(None)
            return

        # PhysX-only: set env id bit count for replicated physics. Newton handles env separation in its own API.
        # Intentionally matches both physx and ovphysx (both are PhysX-based)
        if self.cfg.replicate_physics and "physx" in self.physics_backend:
            prim = self.stage.GetPrimAtPath("/physicsScene")
            prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

        # Suspend Fabric's USD notice listener around bulk authoring. ``restore=False`` because the downstream
        # ``SimulationContext.reset`` does the Fabric resync — re-enabling here would batch-resync everything
        # we just authored, which is slower than the unsuppressed baseline.
        with cloner.disabled_fabric_change_notifies(self.stage, restore=False):
            replicate_args = (plan.sources, plan.destinations, self._ALL_INDICES, plan.clone_mask)

            if not copy_from_source and self.cloner_cfg.physics_clone_fn is not None:
                self.cloner_cfg.physics_clone_fn(
                    self.stage,
                    *replicate_args,
                    positions=self._default_env_origins,
                    device=self.cloner_cfg.device,
                )
            if self.cloner_cfg.clone_usd:
                is_env_root_plan = (
                    len(plan.sources) == 1
                    and plan.sources[0] == self.env_fmt.format(0)
                    and plan.destinations == (self.env_fmt,)
                )
                usd_positions = self._default_env_origins if is_env_root_plan else None
                cloner.usd_replicate(self.stage, *replicate_args, positions=usd_positions)

        # Publish to ``SimulationContext`` (the canonical owner). The :attr:`clone_plan`
        # property below forwards reads back through ``sim.get_clone_plan()`` so consumers
        # holding a scene reference still see the published plan without a duplicate cache.
        self.sim.set_clone_plan(plan)

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
    def clone_plan(self) -> cloner.ClonePlan | None:
        """Clone plan produced by :meth:`clone_environments`.

        Forwards to :meth:`SimulationContext.get_clone_plan`, which is the canonical owner.
        The plan records the source paths, destination templates, and the per-env source
        assignment mask. ``None`` until :meth:`clone_environments` runs.
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
                is_multi_spawner = isinstance(
                    asset_cfg.spawn, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)
                )
                if self.env_ns not in asset_cfg.prim_path:
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
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                    # set spawn_path on spawner if cloning is needed
                    if hasattr(rigid_object_cfg, "spawn") and rigid_object_cfg.spawn is not None:
                        is_multi_spawner = isinstance(
                            rigid_object_cfg.spawn, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)
                        )
                        if self.env_ns not in rigid_object_cfg.prim_path:
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
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    asset_cfg.filter_prim_paths_expr = [
                        p.format(ENV_REGEX_NS=self.env_regex_ns) for p in asset_cfg.filter_prim_paths_expr
                    ]
                    if hasattr(asset_cfg, "sensor_shape_prim_expr") and asset_cfg.sensor_shape_prim_expr:
                        asset_cfg.sensor_shape_prim_expr = [
                            p.format(ENV_REGEX_NS=self.env_regex_ns) for p in asset_cfg.sensor_shape_prim_expr
                        ]
                    if hasattr(asset_cfg, "filter_shape_prim_expr") and asset_cfg.filter_shape_prim_expr:
                        asset_cfg.filter_shape_prim_expr = [
                            p.format(ENV_REGEX_NS=self.env_regex_ns) for p in asset_cfg.filter_shape_prim_expr
                        ]
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
                self._extras[asset_name] = FrameView(asset_cfg.prim_path, device=self.device, stage=self.stage)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")

            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
