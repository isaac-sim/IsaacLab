# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import ModelBuilder

from pxr import Sdf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology
from isaaclab.cloner import path as cloner_path
from isaaclab.physics import PhysicsManager
from isaaclab.scene_data import SceneDataFormat
from isaaclab.scene_data.deformable_discovery import (
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils.queries import has_deformable_body_api

from isaaclab_newton.cloner.newton_clone_utils import (
    add_deformable_from_usd,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonBackendCfg, NewtonCfg, NewtonManager
from isaaclab_newton.sim.spawners.mpm.mpm import _SIMULATION_POINTS_SUFFIX

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


def copy_newton_clone_source(source_path: str, xform: wp.transform | None = None) -> ModelBuilder:
    """Copy a retained clone-source builder without sharing mutable shape geometry.

    Args:
        source_path: Clone-plan source prim path retained during Newton replication.
        xform: Optional transform applied while copying the source.

    Returns:
        An independent builder that is safe to finalize or extend.

    Raises:
        RuntimeError: If Newton replication did not retain the requested source.
    """
    source = NewtonManager._cl_protos.get(source_path)
    if source is None:
        raise RuntimeError(f"No retained Newton clone source for {source_path!r}.")
    builder = ModelBuilder(up_axis=source.up_axis)
    builder.add_builder(source, xform=xform)
    builder.shape_source = [
        value.copy() if callable(getattr(value, "copy", None)) else copy.copy(value) for value in builder.shape_source
    ]
    return builder


@contextlib.contextmanager
def newton_builder_world_hook(
    hook: Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None],
) -> Iterator[None]:
    """Temporarily extend every world built by Newton replication.

    The callback must not already be registered. On exit, the context removes
    only its callback and preserves hooks owned by other callers.

    Args:
        hook: Callback receiving the builder, world index, world position [m] as a NumPy array,
            and world orientation quaternion in xyzw order as a NumPy array during replication.

    Yields:
        Control while the callback is registered.

    Raises:
        RuntimeError: If the callback is already registered.
    """
    hooks = NewtonManager._per_world_builder_hooks
    if hook in hooks:
        raise RuntimeError("Newton world-builder hook is already registered.")
    hooks.append(hook)
    try:
        yield
    finally:
        if hook in hooks:
            hooks.remove(hook)


def _replicate_newton(
    stage: Usd.Stage,
    env_ids: np.ndarray,
    sim: SimulationContext,
    *,
    plan: ClonePlan,
    asset_prototype_ids: Sequence[int],
    positions: np.ndarray | None = None,
    up_axis: str = "Z",
    quaternions: np.ndarray | None = None,
) -> tuple[ModelBuilder, object, dict, NewtonBackendCfg | None]:
    """Import and replicate the plan's Newton representation, with or without Newton physics."""
    cfg = sim.cfg.physics
    sources = cloner_path.get_asset_prototype_paths(plan)
    templates, starts = cloner_path.get_world_prototype_asset_templates(plan)
    shared = templates[: starts[1]]
    global_paths = tuple(
        root for root, parent in zip(shared, cloner_path.get_parent_indices(shared), strict=True) if parent == -1
    )
    # A sensor selects an existing body; opting out of cloning must not remove that body from its owner.
    exclude_paths = tuple(
        source
        for index, source in enumerate(sources)
        if source is not None and index not in asset_prototype_ids
        if not isinstance(plan.asset_cfgs[index], SensorBaseCfg)
    )
    simulation = isinstance(cfg, NewtonCfg)
    if positions is None:
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
    if quaternions is None:
        quaternions = np.zeros((len(env_ids), 4), dtype=np.float32)
        quaternions[:, 3] = 1.0

    manager_cls = sim.physics_manager if simulation else NewtonManager
    schema_resolvers = manager_cls._get_usd_import_schema_resolvers()
    create_builder = partial(manager_cls.create_builder if simulation else ModelBuilder, up_axis=up_axis)
    load_visual_shapes = cfg.load_visual_shapes if simulation else True
    if load_visual_shapes is None:
        load_visual_shapes = sim.is_rendering or sim.can_render_rgb_array() or sim.visual_shapes_required
    builder = create_builder()
    import_paths = (sim.cfg.physics_prim_path, *global_paths) if simulation else global_paths
    source_paths = list(dict.fromkeys(sources[index] for index in asset_prototype_ids if sources[index] is not None))
    if simulation:
        deformable_paths = []
        for source in source_paths:
            prim = stage.GetPrimAtPath(source)
            is_mesh_body = prim and not prim.IsA(UsdGeom.Points) and not prim.IsA(UsdGeom.BasisCurves)
            if is_mesh_body and has_deformable_body_api(prim):
                deformable_paths.append(source)
        ignore_paths = manager_cls._inject_terrain_heightfields(stage, builder, root_paths=import_paths)
        ignore_paths.extend((*exclude_paths, *deformable_paths))
    else:
        entries = deformable_prototypes(stage, plan, exclude_paths=exclude_paths)
        ignore_paths = [*exclude_paths, *(entry.root_path for entry in entries)]

    stage_info = None
    if simulation:
        stage_info = builder.add_usd(stage, root_path=sim.cfg.physics_prim_path, schema_resolvers=schema_resolvers)

    import_results: dict[str, dict[str, Any]] = {}
    options = dict(ignore_paths=ignore_paths, load_visual_shapes=load_visual_shapes)
    options.update(skip_mesh_approximation=not simulation, import_results_out=import_results)
    source_builders = build_source_builders(stage, source_paths, create_builder, schema_resolvers, **options)
    source_geometry = {source: {} for source in source_builders}
    if simulation:
        entries = [add_deformable_from_usd(source_builders[path], stage, root_path=path) for path in deformable_paths]
    else:
        # Import visual meshes once into their owning prototypes, before the common native replication.
        for entry in entries:
            ancestors = reversed(Sdf.Path(entry.root_path).GetPrefixes())
            source = next(str(path) for path in ancestors if str(path) in source_builders)
            native = source_builders[source]
            source_geometry[source][entry.vis_mesh_path] = native.particle_count
            pose = dict(pos=wp.vec3(*entry.init_pos), rot=wp.quat(*entry.init_rot), scale=1.0, vel=wp.vec3())
            if entry.deformable_type == "surface" or entry.vis_mesh_path != entry.sim_mesh_path:
                add_mesh = native.add_cloth_mesh
                mesh = dict(vertices=entry.vis_vertices, indices=entry.vis_indices, density=1.0)
                mesh.update(tri_ke=1e4, tri_ka=1e4, tri_kd=1.5e-6, edge_ke=5.0, edge_kd=1e-2, particle_radius=0.008)
            else:
                add_mesh = native.add_soft_mesh
                mesh = dict(vertices=entry.vertices, indices=entry.indices, density=1000.0)
                mesh.update(k_mu=1e5, k_lambda=1e5, k_damp=0.0)
            add_mesh(**mesh, **pose)

    # Resolve native capsule indices once per source, not by rediscovering labels after cloning.
    source_cables = {}
    for source, imported in import_results.items():
        cables = source_cables[source] = {}
        if not simulation or not imported["path_cable_map"]:
            continue
        shapes = {label: index for index, label in enumerate(source_builders[source].shape_label)}
        for path, (bodies, _) in imported["path_cable_map"].items():
            if imported["path_cable_attrs"][path]["closed"]:
                continue
            if len(UsdGeom.BasisCurves(stage.GetPrimAtPath(path)).GetCurveVertexCountsAttr().Get()) != 1:
                continue
            cables[path] = [shapes[f"{path}_edge_capsule_{segment}"] for segment in range(len(bodies))]
    if simulation:
        global_sites, source_sites, root_sites = NewtonManager._cl_inject_sites(builder, source_builders)
    else:
        # Clear imported filters before merging into a fresh, compact final filter store.
        for imported in source_builders.values():
            imported.shape_collision_filter_pairs = []
            imported.shape_collision_group[:] = [0] * imported.shape_count
        global_sites, source_sites, root_sites = {}, {}, {}

    particle_ranges, visual_ranges, cable_bindings, geometry_offsets = {}, {}, {}, {}
    # The USD importer owns simulation ranges; MPM's spawner authors a separate visible point prim.
    particle_visual_paths = {
        path: path.removesuffix(_SIMULATION_POINTS_SUFFIX) + "/Particles"
        for imported in import_results.values()
        for path in imported["path_particle_map"]
        if path.endswith(_SIMULATION_POINTS_SUFFIX)
        and stage.GetPrimAtPath(path.removesuffix(_SIMULATION_POINTS_SUFFIX) + "/Particles")
    }

    def record_geometry(source: str, destination: str, shape_offset: int, particle_offset: int) -> None:
        for path, shapes in source_cables[source].items():
            cable_bindings[cloner_path.rebase(path, source, destination)] = [shape_offset + shape for shape in shapes]
        for path, (start, end) in import_results[source]["path_particle_map"].items():
            native_range = (particle_offset + start, end - start)
            particle_ranges[cloner_path.rebase(path, source, destination)] = native_range
            if path in particle_visual_paths:
                visual_ranges[cloner_path.rebase(particle_visual_paths[path], source, destination)] = native_range
        for path, offset in source_geometry[source].items():
            geometry_offsets[cloner_path.rebase(path, source, destination)] = particle_offset + offset

    options = dict(env_ids=env_ids, source_site_indices=source_sites, env_root_sites=root_sites)
    options["per_world_builder_hooks"] = NewtonManager._per_world_builder_hooks if simulation else ()
    has_geometry = any(source_cables.values()) or any(result["path_particle_map"] for result in import_results.values())
    options["source_builder_added"] = record_geometry if has_geometry or any(source_geometry.values()) else None
    local_site_map, world_xforms = replicate_builder_mapping(
        builder, plan, positions, quaternions, source_builders, **options
    )
    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    backend_cfg = None
    if simulation:
        NewtonManager._cable_bindings = cable_bindings
        geometry = expand_deformable_entries(entries, plan, env_ids, positions)
        ranges = {
            label: start
            for family in ("cloth", "soft")
            for label, start in zip(
                getattr(builder, f"_{family}_label"), getattr(builder, f"_{family}_particle_start"), strict=True
            )
        }
        offsets = [ranges[entry.root_path] for entry in geometry]
        batches = deformable_geometry_batches(geometry, offsets, device=sim.device)
        if visual_ranges:
            batches.append((SceneDataFormat.Points(), visual_ranges))
        NewtonManager._scene_data_backend._geometry_batches = batches
        NewtonManager._cl_site_index_map = site_index_map
        NewtonManager._world_xforms = world_xforms
        NewtonManager._cl_protos = source_builders
        NewtonManager.set_builder(builder, particle_ranges=particle_ranges)
        NewtonManager._num_envs = len(env_ids)
    else:
        backend_cfg = NewtonBackendCfg(
            builder=builder, device=sim.device, num_envs=len(env_ids), simulation=False, geometry_offsets=geometry_offsets
        )
    return builder, stage_info, site_index_map, backend_cfg


class NewtonReplicateContext:
    """Build one Newton model from the sources routed to it in a clone plan."""

    replicate_priority = 0

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        """Initialize the context from its owning simulation."""
        self._sim = sim_context
        self.up_axis = up_axis
        self.backend_cfg: NewtonBackendCfg | None = None
        """Completed allocation inputs; native resources belong to the simulation registry."""

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> tuple[ModelBuilder, object, dict]:
        """Build and publish a Newton model from this context's source declarations."""
        env_ids = np.arange(len(plan.topology.world_prototype_layout))
        options = dict(plan=plan, asset_prototype_ids=asset_prototype_ids, positions=plan.positions)
        builder, stage_info, sites, self.backend_cfg = _replicate_newton(
            self._sim.stage, env_ids, self._sim, up_axis=self.up_axis, **options
        )
        return builder, stage_info, sites


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    up_axis: str = "Z",
    global_paths: tuple[str, ...] = (),
) -> tuple[ModelBuilder, dict[str, Any]]:
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        global_paths: Shared scene-asset roots imported once. Defaults to none.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    world_masks, first, selected = np.unique(mapping.T, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    members = [np.flatnonzero(mask) for mask in world_masks[order]]
    shared = np.arange(len(sources), len(sources) + len(global_paths))
    prefix, suffix = destinations[0].split("{}", 1) if destinations else ("/World/envs/env_", "")
    topology = PrototypeWorldTopology(
        len(sources) + len(global_paths),
        np.concatenate((shared, *members)).astype(np.int32),
        np.r_[0, len(shared), len(shared) + np.cumsum([len(world) for world in members])],
        np.argsort(order)[selected].astype(np.int32),
    )
    roots = zip((*sources, *global_paths), (*destinations, *global_paths), strict=True)
    assets = tuple(AssetBaseCfg(prim_path=dst.format("[^/]+"), spawn=SpawnerCfg(spawn_path=src)) for src, dst in roots)
    env_template = prefix + "{}" + suffix.split("/", 1)[0]
    plan = ClonePlan(topology, asset_cfgs=assets, env_template=env_template, positions=positions)
    options = dict(plan=plan, asset_prototype_ids=range(len(assets)), positions=positions)
    options.update(up_axis=up_axis, quaternions=quaternions)
    builder, stage_info, _, _ = _replicate_newton(stage, env_ids, PhysicsManager._sim, **options)
    return builder, stage_info
