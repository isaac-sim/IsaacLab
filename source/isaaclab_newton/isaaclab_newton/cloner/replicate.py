# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any

import numpy as np
import warp as wp
from newton import Axis, ModelBuilder, ShapeFlags

from pxr import Sdf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology, ReplicateContext
from isaaclab.cloner import path as cloner_path
from isaaclab.scene_data import SceneDataFormat
from isaaclab.scene_data.deformable_discovery import (
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext, SpawnerCfg

from isaaclab_newton.cloner.newton_clone_utils import (
    add_deformable_from_usd,
    add_visual_deformables_to_sources,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.mesh_variants import _mesh_variant_body_properties, _prepare_mesh_variant, _render_shapes
from isaaclab_newton.sim.spawners.mpm.mpm import _SIMULATION_POINTS_SUFFIX

_PER_WORLD_BUILDER_HOOKS: list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]] = []
"""Callbacks applied to every world built by Newton replication; registered before the simulation exists."""


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
    source = SimulationContext.instance().backend_manager._cl_protos.get(source_path)
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
    hooks = _PER_WORLD_BUILDER_HOOKS
    if hook in hooks:
        raise RuntimeError("Newton world-builder hook is already registered.")
    hooks.append(hook)
    try:
        yield
    finally:
        if hook in hooks:
            hooks.remove(hook)


def _replicate_newton(  # noqa: C901 -- Resolve source ownership and native geometry in one clone pass.
    stage: Usd.Stage,
    env_ids: np.ndarray,
    sim: SimulationContext,
    *,
    plan: ClonePlan,
    asset_prototype_ids: Sequence[int],
    positions: np.ndarray | None = None,
    up_axis: str = "Z",
    quaternions: np.ndarray | None = None,
) -> tuple[ModelBuilder, object, dict]:
    """Import and replicate the plan's Newton representation, with or without Newton physics."""
    cfg = sim.cfg.physics
    sources = cloner_path.get_asset_prototype_paths(plan)
    templates, starts = cloner_path.get_world_prototype_asset_templates(plan)
    shared = templates[: starts[1]]
    global_paths = tuple(
        root for root, parent in zip(shared, cloner_path.get_parent_indices(shared), strict=True) if parent == -1
    )
    routed_sources = tuple(sources[index] for index in asset_prototype_ids if sources[index] is not None)
    world_assets = plan.topology.world_prototypes[starts[1] :]
    has_world_source = any(asset in asset_prototype_ids and sources[asset] is not None for asset in world_assets)
    # A selected source owns its entire USD subtree, including separately declared descendants.
    exclude_paths = tuple(
        source
        for index, source in enumerate(sources)
        if source is not None and index not in asset_prototype_ids
        if not isinstance(plan.asset_cfgs[index], SensorBaseCfg)
        if not any(cloner_path.relative_to(source, owner) is not None for owner in routed_sources)
    )
    simulation = isinstance(cfg, NewtonCfg)
    manager = sim.backend_manager if simulation else NewtonManager
    catalog_rows = {}
    if simulation and manager._supports_mesh_variants and not cfg.solver_cfg.use_mujoco_contacts:
        names = dict.fromkeys(asset.prim_path for asset in plan.asset_cfgs if isinstance(asset, RigidObjectCfg))
        for name in names:
            rows = tuple(cloner_path.get_asset_prototypes(plan, name))
            if len(rows) < 2 or any(sources[row] is None for row in rows):
                continue
            members = [
                [
                    index
                    for index in range(starts[prototype + 1], starts[prototype + 2])
                    if plan.topology.world_prototypes[index] in rows
                ]
                for prototype in plan.topology.world_prototype_layout
            ]
            if all(len(indices) == 1 for indices in members):
                catalog_rows[rows] = [indices[0] for indices in members]
    variant_sources = tuple(dict.fromkeys(sources[row] for rows in catalog_rows for row in rows))
    if positions is None:
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
    if quaternions is None:
        quaternions = np.zeros((len(env_ids), 4), dtype=np.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = manager._get_usd_import_schema_resolvers()
    create_builder = partial(manager.create_builder if simulation else ModelBuilder, up_axis=up_axis)
    load_visual_shapes = cfg.load_visual_shapes if simulation else True
    if load_visual_shapes is None:
        load_visual_shapes = sim.is_rendering or sim.can_render_rgb_array() or sim.visual_shapes_required
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg))
    builder.up_axis = Axis.from_string(up_axis)
    import_paths = (sim.cfg.physics_prim_path, *global_paths) if simulation else global_paths
    # A parent import owns its whole subtree; reject nested replacements before importing either source.
    needed = set()
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        references = []
        for index in range(start, end):
            if (asset := plan.topology.world_prototypes[index]) in asset_prototype_ids and sources[asset] is not None:
                references.append((sources[asset], templates[index]))
        parents = cloner_path.get_parent_indices([target for _, target in references])
        for (source, target), parent in zip(references, parents, strict=True):
            if parent == -1:
                # Shared imports cannot own the namespace where separate environment sources are cloned.
                shared_owns_env = end == starts[1] and cloner_path.relative_to(plan.env_template, target) is not None
                if has_world_source and shared_owns_env:
                    raise ValueError(
                        f"Shared Newton source {source!r} at {target!r} overlaps the environment namespace "
                        f"{plan.env_template!r}; declare individual global assets outside it."
                    )
                needed.add(source)
                continue
            parent_source, parent_target = references[parent]
            expected = cloner_path.rebase(target, parent_target, parent_source)
            if source != expected:
                raise ValueError(
                    f"Cannot clone {source!r} to {target!r}: {parent_target!r} already owns that subtree "
                    f"from {parent_source!r}. A nested Newton source must be {expected!r}."
                )
    source_paths = list(dict.fromkeys([*(source for source in routed_sources if source in needed), *variant_sources]))
    # A parent source also owns deformables declared beneath it, even when the child has its own asset config.
    entries = deformable_prototypes(stage, plan, exclude_paths=exclude_paths)
    ignore_paths = [*exclude_paths, *(entry.root_path for entry in entries)]
    stage_info = None
    if simulation:
        ignore_paths.extend(manager._inject_terrain_heightfields(stage, builder, root_paths=import_paths))
        stage_info = builder.add_usd(stage, root_path=sim.cfg.physics_prim_path, schema_resolvers=schema_resolvers)

    ignore_paths = [path for path in ignore_paths if path not in variant_sources]

    import_results: dict[str, dict[str, Any]] = {}
    options = dict(ignore_paths=ignore_paths, load_visual_shapes=load_visual_shapes)
    options.update(skip_mesh_approximation=not simulation, import_results_out=import_results)
    source_builders = build_source_builders(stage, source_paths, create_builder, schema_resolvers, **options)
    catalogs = {}
    target_owners = {}
    for rows, members in catalog_rows.items():
        builders = tuple(source_builders[sources[row]] for row in rows)
        properties = _mesh_variant_body_properties(
            builders, plan.asset_cfgs[rows[0]].mesh_variant_inertia_diagonal_offset
        )
        if properties is None:
            continue
        for row, imported in zip(rows, builders, strict=True):
            if cloner_path.relative_to(imported.body_label[0], sources[row]) is None:
                raise ValueError(f"Mesh variant body {imported.body_label[0]!r} is outside source {sources[row]!r}.")
        variant_by_asset = {row: variant for variant, row in enumerate(rows)}
        initial_ids, targets = [], []
        for world, member in enumerate(members):
            row = plan.topology.world_prototypes[member]
            target = cloner_path.rebase(
                source_builders[sources[row]].body_label[0], sources[row], templates[member].format(world)
            )
            initial_ids.append(variant_by_asset[row])
            targets.append(target)
            target_owners.setdefault(target, []).append(rows)
        catalogs[rows] = properties, np.asarray(initial_ids, dtype=np.int32), targets
    # Distinct asset declarations can be views of the same body; neither can own an independent switch.
    conflicts = {rows for owners in target_owners.values() if len(owners) > 1 for rows in owners}
    catalogs = {rows: properties for rows, properties in catalogs.items() if rows not in conflicts}
    if load_visual_shapes:
        # Native source imports are shared: stripping visuals must not alter a non-switchable asset.
        while catalogs:
            owned = {row for rows in catalogs for row in rows}
            shared = {source for row, source in enumerate(sources) if row not in owned}
            exclusive = {
                rows: props for rows, props in catalogs.items() if not any(sources[row] in shared for row in rows)
            }
            if len(exclusive) == len(catalogs):
                break
            catalogs = exclusive
    variant_sources = tuple(dict.fromkeys(sources[row] for rows in catalogs for row in rows))
    variant_visual_shapes = {}
    if simulation and load_visual_shapes and variant_sources:
        # Collision topology stays strided; ragged visual geometry is a separate renderer projection.
        variant_visual_shapes = {
            source: _render_shapes(source_builders[source], ShapeFlags.VISIBLE) for source in variant_sources
        }
        collision_sources = build_source_builders(
            stage,
            variant_sources,
            create_builder,
            schema_resolvers,
            ignore_paths=ignore_paths,
            load_visual_shapes=False,
        )
        for source in collision_sources.values():
            source.shape_flags = [flags & ~ShapeFlags.VISIBLE for flags in source.shape_flags]
        source_builders.update(collision_sources)
    # Imports contain authored USD poses, which need not include the planned world placement.
    source_inverse = {}
    xforms = UsdGeom.XformCache()
    for source in source_builders:
        transform = wp.transform()
        if (match := cloner_path.match(source, plan.env_template)) is not None:
            anchor = stage.GetPrimAtPath(plan.env_template.format(match.instance))
            frame = xforms.GetLocalToWorldTransform(anchor).RemoveScaleShear().GetInverse()
            rotation = frame.ExtractRotationQuat()
            transform = wp.transform(frame.ExtractTranslation(), (*rotation.GetImaginary(), rotation.GetReal()))
        source_inverse[source] = np.asarray(transform, dtype=np.float32)
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
        for entry in entries:
            ancestors = reversed(Sdf.Path(entry.root_path).GetPrefixes())
            owners = [source_builders[str(path)] for path in ancestors if str(path) in source_builders]
            if not owners:
                raise RuntimeError(f"No imported source owns deformable {entry.root_path!r}.")
            for source in owners:
                add_deformable_from_usd(source, stage, entry)
        global_sites, source_sites, root_sites = sim.backend_manager._cl_inject_sites(builder, source_builders)
    else:
        add_visual_deformables_to_sources(source_builders, entries)
        # Clear imported filters before merging into a fresh, compact final filter store.
        for imported in source_builders.values():
            imported.shape_collision_filter_pairs = []
            imported.shape_collision_group[:] = [0] * imported.shape_count
        global_sites, source_sites, root_sites = {}, {}, {}

    particle_ranges, visual_ranges, cable_bindings = {}, {}, {}
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

    options = dict(env_ids=env_ids, source_site_indices=source_sites, env_root_sites=root_sites)
    options["per_world_builder_hooks"] = _PER_WORLD_BUILDER_HOOKS if simulation else ()
    has_geometry = any(source_cables.values()) or any(result["path_particle_map"] for result in import_results.values())
    options["source_builder_added"] = record_geometry if has_geometry else None
    local_site_map, world_xforms = replicate_builder_mapping(
        builder, plan, positions, quaternions, source_builders, source_inverse=source_inverse, **options
    )
    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    if simulation:
        manager._cable_bindings = cable_bindings
        geometry = expand_deformable_entries(entries, plan, env_ids, positions, imported_sources=source_builders)
        ranges = dict(zip(builder.surface_label, builder._surface_particle_start, strict=True))
        ranges.update(zip(builder.volume_label, builder._volume_particle_start, strict=True))
        offsets = [ranges[entry.root_path] for entry in geometry]
        batches = deformable_geometry_batches(geometry, offsets, device=sim.device)
        if visual_ranges:
            batches.append((SceneDataFormat.Points(), visual_ranges))
        manager._scene_data_backend._geometry_batches = batches
        manager._cl_site_index_map = site_index_map
        manager._world_xforms = world_xforms
        manager._cl_protos = source_builders
        manager._particle_ranges = particle_ranges
        manager._num_envs = len(env_ids)
        if catalogs:
            bodies = {label: index for index, label in enumerate(builder.body_label)}
            builder.mesh_variant_definitions = {}
            for rows, (properties, initial_ids, targets) in catalogs.items():
                shapes = [
                    [
                        shape
                        for shape in builder.body_shapes[bodies[target]]
                        if int(builder.shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)
                    ]
                    for target in targets
                ]
                sources_for_catalog = tuple(source_builders[sources[row]] for row in rows)
                builder.mesh_variant_definitions[plan.asset_cfgs[rows[0]].prim_path] = _prepare_mesh_variant(
                    builder,
                    sources_for_catalog,
                    np.asarray(shapes, dtype=np.int32),
                    properties,
                    initial_ids,
                    tuple(variant_visual_shapes[sources[row]] for row in rows) if variant_visual_shapes else None,
                )
    return builder, stage_info, site_index_map


class NewtonReplicateContext(ReplicateContext):
    """Build one Newton model from the sources routed to it in a clone plan."""

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        """Initialize the context from its owning simulation."""
        super().__init__(sim_context)
        self.up_axis = up_axis

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> tuple[ModelBuilder, object, dict]:
        """Populate the shared Newton builder from this context's source declarations."""
        env_ids = np.arange(len(plan.topology.world_prototype_layout))
        options = dict(plan=plan, asset_prototype_ids=asset_prototype_ids, positions=plan.positions)
        return _replicate_newton(self._sim.stage, env_ids, self._sim, up_axis=self.up_axis, **options)


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
    builder, stage_info, _ = _replicate_newton(stage, env_ids, SimulationContext.instance(), **options)
    return builder, stage_info
