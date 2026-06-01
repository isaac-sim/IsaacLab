# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Model, ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverVBD

from isaaclab.sim.utils.stage import get_current_stage

from .deformable_object import (
    add_deformable_entry_to_builder,
    clear_deformable_builder_hooks,
    install_deformable_builder_hooks,
)
from .newton_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class NewtonVBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the VBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.

        TODO: Subclass should not override this method, once deformables
        supported on Newton import_usd, this can be unified with NewtonManager's
        implementation.
        """

        # Deformable body registry and extension hooks.
        # Experimental deformable support registers callbacks here so the manager
        # and cloner can invoke them without hard-coding deformable logic.
        install_deformable_builder_hooks()

        super().initialize(sim_context)

    @classmethod
    def _solver_specific_clear(cls):
        """Clear VBD-specific state."""
        clear_deformable_builder_hooks()

    @classmethod
    def _get_deformable_ignore_paths(cls) -> list[str]:
        """Return USD prim paths to skip when calling ``builder.add_usd``.

        For each registered deformable body, both the simulation mesh (which
        carries ``UsdPhysics.CollisionAPI``) and the visual mesh are returned.
        The sim mesh must be skipped so Newton does not create a redundant
        static mesh collider alongside the particles produced by
        ``add_soft_mesh``.  The visual mesh is skipped so Newton does not
        treat it as a collider — Kit reads it directly from USD for rendering.

        Paths may contain regex patterns; Newton's ``add_usd`` matches them
        via :func:`re.match`.
        """
        paths: list[str] = []
        for entry in cls._deformable_registry:
            paths.append(entry.sim_mesh_prim_path)
            paths.append(entry.vis_mesh_prim_path)
        return paths

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.

        TODO: Subclass should not override this method, missing piece is
        having Newton bind a surface mesh to volume deformable tetrahedral mesh
        in addition to removing the deformable_registry data structure.
        """
        super().start_simulation()

        # Apply global model parameters from :class:`NewtonModelCfg` to the finalized model.
        # Sets ``soft_contact_ke/kd/mu`` and optionally overrides per-shape
        # ``shape_material_ke/kd/mu`` on the Newton model.
        from isaaclab.physics import PhysicsManager

        cfg = PhysicsManager._cfg
        if cfg is not None and hasattr(cfg, "model_cfg") and cfg.model_cfg is not None:
            model = cls._model
            if model is None:
                return

            model_cfg = cfg.model_cfg
            model.soft_contact_ke = float(model_cfg.soft_contact_ke)
            model.soft_contact_kd = float(model_cfg.soft_contact_kd)
            model.soft_contact_mu = float(model_cfg.soft_contact_mu)

            if model_cfg.shape_material_ke is not None:
                model.shape_material_ke.fill_(float(model_cfg.shape_material_ke))
            if model_cfg.shape_material_kd is not None:
                model.shape_material_kd.fill_(float(model_cfg.shape_material_kd))
            if model_cfg.shape_material_mu is not None:
                model.shape_material_mu.fill_(float(model_cfg.shape_material_mu))

        # Setup USD/Fabric sync for Kit viewport deformable rendering
        if not cls._clone_physics_only and cls._deformable_registry:
            import re

            import usdrt

            if NewtonManager._usdrt_stage is None:
                NewtonManager._usdrt_stage = get_current_stage(fabric=True)

            stage = get_current_stage()
            for entry in cls._deformable_registry:
                for inst_idx, offset in enumerate(entry.particle_offsets):
                    # Resolve regex pattern to concrete instance path of visual mesh
                    resolved_vis = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), entry.vis_mesh_prim_path)
                    resolved_vis = re.sub(r"\.\*", str(inst_idx), resolved_vis)
                    vis_prim = stage.GetPrimAtPath(resolved_vis)

                    if not vis_prim or not vis_prim.IsValid():
                        logger.warning("[setup_fabric_particle_sync] vis prim not found at %s", resolved_vis)
                        continue

                    # Create per-instance particle offset and count attributes on the visual mesh
                    # prim so the Fabric sync kernel can find the right slice of particle_q
                    # and iterate only over this body's particles (counts vary across bodies).
                    fab_prim = NewtonManager._usdrt_stage.GetPrimAtPath(vis_prim.GetPath().pathString)
                    fab_prim.CreateAttribute(
                        NewtonManager._newton_particle_offset_attr, usdrt.Sdf.ValueTypeNames.UInt, True
                    )
                    fab_prim.GetAttribute(NewtonManager._newton_particle_offset_attr).Set(offset)
                    fab_prim.CreateAttribute(
                        NewtonManager._newton_particle_count_attr, usdrt.Sdf.ValueTypeNames.UInt, True
                    )
                    fab_prim.GetAttribute(NewtonManager._newton_particle_count_attr).Set(entry.particles_per_body)

            cls._mark_particles_dirty()
            cls.sync_particles_to_usd()

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Create builder from USD stage with special treatment for deformable
        bodies, as these are not read from USD yet.

        Detects env Xforms (e.g. ``/World/Env_0``, ``/World/Env_1``) and builds
        each as a separate Newton world via ``begin_world``/``end_world``.
        Falls back to a flat ``add_usd`` when no env Xforms are found.

        TODO: Subclass should not override this method, once deformables
        supported on Newton import_usd, this can be unified with NewtonManager's
        implementation.
        """
        import re

        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)

        # Scan /World children for env-like Xforms (Env_0, env_1, ...)
        env_pattern = re.compile(r"^[Ee]nv_(\d+)$")
        world_prim = stage.GetPrimAtPath("/World")
        env_paths: list[tuple[int, str]] = []
        if world_prim and world_prim.IsValid():
            for child in world_prim.GetChildren():
                m = env_pattern.match(child.GetName())
                if m:
                    env_paths.append((int(m.group(1)), child.GetPath().pathString))
        env_paths.sort(key=lambda x: x[0])

        builder = ModelBuilder(up_axis=up_axis)

        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

        # Deformable sim/visual mesh paths must be skipped by ``add_usd``
        # so they don't get duplicated as static colliders.
        deformable_ignore_paths = cls._get_deformable_ignore_paths()

        if not env_paths:
            # No env Xforms — flat loading
            builder.add_usd(stage, ignore_paths=deformable_ignore_paths, schema_resolvers=schema_resolvers)

            # Add deformable bodies from the registry (single world at origin).
            for entry in cls._deformable_registry:
                add_deformable_entry_to_builder(builder, entry, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        else:
            # Load everything except the env subtrees (ground plane, lights, etc.)
            ignore_paths = [path for _, path in env_paths] + deformable_ignore_paths
            builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)

            # Build a prototype from the first env (all envs assumed identical)
            _, proto_path = env_paths[0]
            proto = ModelBuilder(up_axis=up_axis)
            proto.add_usd(
                stage,
                root_path=proto_path,
                ignore_paths=deformable_ignore_paths,
                schema_resolvers=schema_resolvers,
            )

            # Inject registered sites into the proto before replication
            global_sites, proto_sites, world_sites = cls._cl_inject_sites(builder, {proto_path: proto})
            global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}
            num_worlds = len(env_paths)
            local_site_map: dict[str, list[list[int]]] = {}
            site_entries = proto_sites.get(id(proto), {})

            # Add each env as a separate Newton world
            xform_cache = UsdGeom.XformCache()
            for col, (_, env_path) in enumerate(env_paths):
                builder.begin_world()
                offset = builder.shape_count
                world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
                translation = world_xform.ExtractTranslation()
                rotation = world_xform.ExtractRotationQuat()
                pos = (translation[0], translation[1], translation[2])
                quat = (
                    rotation.GetImaginary()[0],
                    rotation.GetImaginary()[1],
                    rotation.GetImaginary()[2],
                    rotation.GetReal(),
                )
                env_xform = wp.transform(pos, quat)
                builder.add_builder(proto, xform=env_xform)
                for label, xform in world_sites.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(env_xform, xform), label=label)
                    local_site_map[label][col].append(site_idx)
                for label, proto_shape_indices in site_entries.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    for proto_shape_idx in proto_shape_indices:
                        local_site_map[label][col].append(offset + proto_shape_idx)

                # Add deformable bodies from the registry into this world.
                for entry in cls._deformable_registry:
                    add_deformable_entry_to_builder(builder, entry, col, list(pos), quat)

                builder.end_world()

            NewtonManager._cl_site_index_map = {
                **global_site_map,
                **{label: (None, per_world) for label, per_world in local_site_map.items()},
            }
            NewtonManager._num_envs = len(env_paths)

        # Call builder.color() if any deformable entries were added (required by VBD solver)
        if cls._deformable_registry:
            builder.color()

        cls.set_builder(builder)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct :class:`SolverVBD` and populate the base-class slots.

        VBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        valid = set(inspect.signature(SolverVBD.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        NewtonManager._solver = SolverVBD(model, **kwargs)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild BVH once per step for solvers that require it (e.g. VBD cloth).
        if hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
