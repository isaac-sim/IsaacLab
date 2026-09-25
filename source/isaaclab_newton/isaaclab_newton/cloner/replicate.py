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
from isaaclab.cloner import ClonePlan
from isaaclab.cloner.path import rebase, under
from isaaclab.cloner.query import iter_sources
from isaaclab.physics import PhysicsManager
from isaaclab.scene_data import SceneDataFormat
from isaaclab.scene_data.deformable_discovery import (
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors
from isaaclab.sim.utils.queries import has_deformable_body_api

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    add_deformable_from_usd,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonBackendCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder
from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths
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
    if xform is None:
        builder.add_builder(source)
    else:
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
    plan: ClonePlan,
    rows: tuple[int, ...],
    sim: SimulationContext,
    *,
    up_axis: str = "Z",
    quaternions: np.ndarray | None = None,
) -> tuple[ModelBuilder, object, dict, NewtonBackendCfg | None]:
    """Import and replicate the plan's Newton representation, with or without Newton physics."""
    cfg = sim.cfg.physics
    simulation = isinstance(cfg, NewtonCfg)
    sources = tuple(plan.sources[row] for row in rows)
    positions = plan.positions
    if positions is None:
        positions = np.zeros((len(plan.env_ids), 3), dtype=np.float32)
    if quaternions is None:
        quaternions = np.zeros((len(plan.env_ids), 4), dtype=np.float32)
        quaternions[:, 3] = 1.0

    manager_cls = sim.physics_manager if simulation else NewtonManager
    schema_resolvers = manager_cls._get_usd_import_schema_resolvers()
    create_builder = partial(manager_cls.create_builder if simulation else ModelBuilder, up_axis=up_axis)
    load_visual_shapes = cfg.load_visual_shapes if simulation else True
    if load_visual_shapes is None:
        load_visual_shapes = sim.is_rendering or sim.can_render_rgb_array() or sim.visual_shapes_required
    builder = create_builder()
    import_paths = (sim.cfg.physics_prim_path, *plan.global_paths) if simulation else plan.global_paths
    if simulation:
        paths = set(sources) | set(plan.global_paths)
        for asset_cfg in plan.cfgs:
            if not isinstance(asset_cfg, AssetBaseCfg):
                continue
            path = asset_cfg.prim_path
            paths.update(source_path for source, _, source_path, _ in iter_sources(plan, path) if source in sources)
            if any(under(path, root) for root in (*sources, *plan.global_paths)):
                paths.add(path)
        deformable_paths = [
            path
            for path in sorted(paths)
            if (prim := stage.GetPrimAtPath(path))
            and has_deformable_body_api(prim)
            and not prim.IsA(UsdGeom.Points)
            and not prim.IsA(UsdGeom.BasisCurves)
        ]
        ignore_paths = list(deformable_paths)
        global_ignore_paths = manager_cls._inject_terrain_heightfields(stage, builder, root_paths=import_paths)
        global_ignore_paths.extend((*plan.sources, *ignore_paths))
    else:
        entries = deformable_prototypes(stage, plan, rows)
        ignore_paths = [entry.root_path for entry in entries]
        global_ignore_paths = [*plan.sources, *ignore_paths]

    import_results = []
    for root_path in import_paths:
        import_result = builder.add_usd(
            stage,
            root_path=root_path,
            ignore_paths=global_ignore_paths,
            schema_resolvers=schema_resolvers,
            load_visual_shapes=load_visual_shapes,
            skip_mesh_approximation=not simulation,
            return_deformable_results=True,
        )
        _restore_visible_colliders_without_visual_shapes(
            builder, stage, import_result["path_shape_map"], load_visual_shapes
        )
        import_results.append(import_result)
    if simulation:
        replace_newton_builder_shape_colors(builder, stage)
    if load_visual_shapes:
        import_builder_visual_material_paths(builder, stage)

    source_import_results: dict[str, dict[str, Any]] = {}
    source_builders = build_source_builders(
        stage,
        sources,
        create_builder,
        schema_resolvers,
        ignore_paths=ignore_paths,
        load_visual_shapes=load_visual_shapes,
        skip_mesh_approximation=not simulation,
        import_results_out=source_import_results,
    )
    if simulation:
        entries = []
        for path in deformable_paths:
            owner = Sdf.Path(path)
            while owner != Sdf.Path.absoluteRootPath and str(owner) not in source_builders:
                owner = owner.GetParentPath()
            source_builder = builder if owner == Sdf.Path.absoluteRootPath else source_builders[str(owner)]
            entries.append(add_deformable_from_usd(source_builder, stage, root_path=path))

    if simulation:
        global_sites, source_sites, root_sites = NewtonManager._cl_inject_sites(builder, source_builders)
    else:
        # Clear imported filters before merging into a fresh, compact final filter store.
        global_builder = builder
        for imported in (global_builder, *source_builders.values()):
            imported.shape_collision_filter_pairs = []
            imported.shape_collision_group[:] = [0] * imported.shape_count
        builder = create_builder()
        builder.add_builder(global_builder)
        global_sites, source_sites, root_sites = {}, {}, {}

    particle_ranges = {
        path: (start, end - start)
        for imported in import_results
        for path, (start, end) in imported["path_particle_map"].items()
    }
    # MPM's importer supplies native ranges; its spawner authors a separate visible point prim.
    particle_visual_paths = {
        path: path.removesuffix(_SIMULATION_POINTS_SUFFIX) + "/Particles"
        for imported in (*import_results, *source_import_results.values())
        for path in imported["path_particle_map"]
        if path.endswith(_SIMULATION_POINTS_SUFFIX)
        and stage.GetPrimAtPath(path.removesuffix(_SIMULATION_POINTS_SUFFIX) + "/Particles")
    }
    visual_ranges = {
        particle_visual_paths[path]: value for path, value in particle_ranges.items() if path in particle_visual_paths
    }

    def record_source_particle_ranges(source: str, destination: str, particle_offset: int) -> None:
        for path, (start, end) in source_import_results[source]["path_particle_map"].items():
            particle_ranges[rebase(path, source, destination)] = (particle_offset + start, end - start)
            if path in particle_visual_paths:
                visual_ranges[rebase(particle_visual_paths[path], source, destination)] = (
                    particle_offset + start,
                    end - start,
                )

    local_site_map, world_xforms = replicate_builder_mapping(
        builder=builder,
        sources=sources,
        mapping=plan.clone_mask[list(rows)],
        positions=positions,
        quaternions=quaternions,
        source_builders=source_builders,
        destinations=tuple(plan.destinations[row] for row in rows),
        env_ids=plan.env_ids,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=NewtonManager._per_world_builder_hooks if simulation else (),
        source_builder_added=record_source_particle_ranges
        if any(imported["path_particle_map"] for imported in source_import_results.values())
        else None,
    )
    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    backend_cfg = None
    if simulation:
        # Only the physics representation publishes cable geometry to SDP.
        cable_counts = {}
        for source, imported in [(None, result) for result in import_results] + list(source_import_results.items()):
            for path, (bodies, _) in imported["path_cable_map"].items():
                if imported["path_cable_attrs"][path]["closed"]:
                    continue
                if len(UsdGeom.BasisCurves(stage.GetPrimAtPath(path)).GetCurveVertexCountsAttr().Get()) != 1:
                    continue
                if source is None:
                    cable_counts[path] = len(bodies)
                else:
                    for row in rows:
                        if plan.sources[row] == source:
                            for column in np.flatnonzero(plan.clone_mask[row]):
                                destination = plan.destinations[row].format(int(plan.env_ids[column]))
                                cable_counts[rebase(path, source, destination)] = len(bodies)
        shape_ids = {label: index for index, label in enumerate(builder.shape_label)} if cable_counts else {}
        NewtonManager._cable_bindings = {
            path: [shape_ids[f"{path}_edge_capsule_{segment}"] for segment in range(count)]
            for path, count in cable_counts.items()
        }
        geometry = expand_deformable_entries(plan, entries, rows)
        ranges = {
            label: start
            for family in ("cloth", "soft")
            for label, start in zip(
                getattr(builder, f"_{family}_label"), getattr(builder, f"_{family}_particle_start"), strict=True
            )
        }
        NewtonManager._scene_data_backend._geometry_batches = deformable_geometry_batches(
            geometry, [ranges[entry.root_path] for entry in geometry], device=sim.device
        )
        if visual_ranges:
            NewtonManager._scene_data_backend._geometry_batches.append((SceneDataFormat.Points(), visual_ranges))
        NewtonManager._cl_site_index_map = site_index_map
        NewtonManager._world_xforms = world_xforms
        NewtonManager._cl_protos = source_builders
        NewtonManager.set_builder(builder, particle_ranges=particle_ranges)
        NewtonManager._num_envs = len(plan.env_ids)
    else:
        geometry_offsets = add_shadow_deformables_to_builder(builder, expand_deformable_entries(plan, entries, rows))
        backend_cfg = NewtonBackendCfg(
            builder=builder,
            device=sim.device,
            num_envs=len(plan.env_ids),
            simulation=False,
            geometry_offsets=geometry_offsets,
        )
    return builder, import_results[0] if simulation else None, site_index_map, backend_cfg


class NewtonReplicateContext:
    """Build one Newton model from the rows routed to it in a clone plan."""

    replicate_priority = 0

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        """Initialize the context from its owning simulation."""
        self._sim = sim_context
        self.up_axis = up_axis
        self.backend_cfg: NewtonBackendCfg | None = None
        """Completed allocation inputs; native resources belong to the simulation registry."""

    def replicate(self, plan: ClonePlan) -> tuple[ModelBuilder, object, dict]:
        """Build and publish a Newton model from this context's plan rows."""
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required for replication.")
        builder, stage_info, sites, self.backend_cfg = _replicate_newton(
            self._sim.stage, plan, plan.context_rows[type(self)], self._sim, up_axis=self.up_axis
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
    *,
    cfgs: tuple[Any, ...] = (),
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
        cfgs: Asset declarations within the sources, including deformables imported with Newton materials.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    plan = ClonePlan(
        sources=tuple(sources),
        destinations=tuple(destinations),
        env_ids=env_ids,
        clone_mask=mapping,
        positions=positions,
        global_paths=global_paths,
        cfgs=cfgs,
    )
    builder, stage_info, _, _ = _replicate_newton(
        stage,
        plan,
        tuple(range(len(sources))),
        PhysicsManager._sim,
        up_axis=up_axis,
        quaternions=quaternions,
    )
    return builder, stage_info
