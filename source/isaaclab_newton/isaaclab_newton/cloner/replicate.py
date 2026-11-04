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
from newton import Axis, ModelBuilder

from pxr import Sdf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology, ReplicateContext
from isaaclab.cloner import path as cloner_path
from isaaclab.cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE
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
from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg, NewtonCloneRecord
from isaaclab_newton.physics.newton_builder import create_builder, inject_terrain_heightfields, usd_schema_resolvers
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
    source = SimulationContext.instance().backend_manager.get_clone_source_builders().get(source_path)
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
    if positions is None:
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
    if quaternions is None:
        quaternions = np.zeros((len(env_ids), 4), dtype=np.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = usd_schema_resolvers(cfg.solver_cfg if simulation else None)
    builder_factory = partial(create_builder, cfg) if simulation else ModelBuilder
    builder_factory = partial(builder_factory, up_axis=up_axis)
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
    source_paths = list(dict.fromkeys(source for source in routed_sources if source in needed))
    # A parent source also owns deformables declared beneath it, even when the child has its own asset config.
    entries = deformable_prototypes(stage, plan, exclude_paths=exclude_paths)
    ignore_paths = [*exclude_paths, *(entry.root_path for entry in entries)]
    stage_info = None
    if simulation:
        ignore_paths.extend(inject_terrain_heightfields(stage, builder, root_paths=import_paths, device=sim.device))
        stage_info = builder.add_usd(
            stage,
            root_path=sim.cfg.physics_prim_path,
            schema_resolvers=schema_resolvers,
            allow_unsafe_torch_checkpoints=True,
        )

    import_results: dict[str, dict[str, Any]] = {}
    options = dict(ignore_paths=ignore_paths, load_visual_shapes=load_visual_shapes)
    options.update(skip_mesh_approximation=not simulation, import_results_out=import_results)
    source_builders = build_source_builders(stage, source_paths, builder_factory, schema_resolvers, **options)
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
        global_sites, source_sites, root_sites = sim.backend_manager.inject_sites(builder, source_builders)
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
    world_hooks = _PER_WORLD_BUILDER_HOOKS if simulation else ()
    options["per_world_builder_hooks"] = world_hooks
    has_geometry = any(source_cables.values()) or any(result["path_particle_map"] for result in import_results.values())
    options["source_builder_added"] = record_geometry if has_geometry else None
    local_site_map, world_xforms = replicate_builder_mapping(
        builder, plan, positions, quaternions, source_builders, source_inverse=source_inverse, **options
    )
    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    if simulation:
        geometry = expand_deformable_entries(entries, plan, env_ids, positions, imported_sources=source_builders)
        ranges = dict(zip(builder.surface_label, builder._surface_particle_start, strict=True))
        ranges.update(zip(builder.volume_label, builder._volume_particle_start, strict=True))
        offsets = [ranges[entry.root_path] for entry in geometry]
        batches = deformable_geometry_batches(geometry, offsets, device=sim.device)
        if visual_ranges:
            batches.append((SceneDataFormat.Points(), visual_ranges))
        record = NewtonCloneRecord(
            world_xforms=world_xforms,
            source_builders=source_builders,
            particle_ranges=particle_ranges,
            cable_bindings=cable_bindings,
            geometry_batches=batches,
        )
        sim.backend_manager.record_clone(record, site_index_map)
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
    prefix, suffix = (destinations[0] if destinations else DEFAULT_ENV_TEMPLATE).split("{}", 1)
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
