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

from pxr import Usd, UsdGeom

from isaaclab.cloner import ClonePlan, PrototypeWorldTopology
from isaaclab.cloner import path as cloner_path
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data.deformable_discovery import deformable_prototypes, expand_deformable_entries

from isaaclab_newton.cloner.newton_clone_utils import (
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonBackendCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder

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
    env_ids: np.ndarray,
    sim: SimulationContext,
    *,
    plan: ClonePlan,
    instances: tuple,
    reference_instances: tuple,
    positions: np.ndarray | None = None,
    global_paths: Sequence[str] = (),
    exclude_paths: Sequence[str] = (),
    up_axis: str = "Z",
    quaternions: np.ndarray | None = None,
) -> tuple[ModelBuilder, object, dict]:
    """Import and replicate the plan's Newton representation, with or without Newton physics."""
    # MPMObject imports NewtonManager, so defer this reciprocal import until model construction.
    from isaaclab_newton.assets.mpm_object.mpm_object import (  # noqa: PLC0415
        record_registered_mpm_particle_ranges,
        reset_registered_mpm_particle_ranges,
    )

    cfg = sim.cfg.physics
    simulation = isinstance(cfg, NewtonCfg)
    if simulation:
        reset_registered_mpm_particle_ranges()
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
    if simulation:
        global_ignore_paths = manager_cls._inject_terrain_heightfields(stage, builder, root_paths=import_paths)
        # Native deformables are appended by per-world hooks, not the USD importer.
        ignore_paths = [
            source + matched.suffix
            for entry in NewtonManager._deformable_registry
            for _, source, destination, world_ids in instances
            if len(world_ids) and world_ids[0] != -1
            if (matched := cloner_path.match(entry.prim_path, destination)) is not None
        ]
        global_ignore_paths.extend(ignore_paths)
        global_ignore_paths.extend(entry.prim_path for entry in NewtonManager._deformable_registry)
    else:
        entries = deformable_prototypes(stage, reference_instances, global_paths, exclude_paths=exclude_paths)
        ignore_paths = list(
            dict.fromkeys(
                path for entry in entries for path in (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path)
            )
        )
        global_ignore_paths = [*exclude_paths, *ignore_paths]

    stage_info = None
    if simulation:
        stage_info = builder.add_usd(stage, root_path=sim.cfg.physics_prim_path, schema_resolvers=schema_resolvers)

    source_import_results: dict[str, dict[str, Any]] = {}
    source_builders = build_source_builders(
        stage,
        tuple(dict.fromkeys(source for _, source, _, world_ids in instances if len(world_ids))),
        create_builder,
        schema_resolvers,
        ignore_paths=global_ignore_paths,
        load_visual_shapes=load_visual_shapes,
        skip_mesh_approximation=not simulation,
        import_results_out=source_import_results,
    )

    # Keep only renderable cables from this representation's actual imports.
    cable_counts = {}
    for source, imported in source_import_results.items():
        for path, (bodies, _) in imported["path_cable_map"].items():
            if imported["path_cable_attrs"][path]["closed"]:
                continue
            if len(UsdGeom.BasisCurves(stage.GetPrimAtPath(path)).GetCurveVertexCountsAttr().Get()) != 1:
                continue
            for _, prototype, destination, world_ids in instances:
                if prototype == source:
                    for world_id in world_ids:
                        target = destination.format(-1 if world_id == -1 else int(env_ids[world_id]))
                        cable_counts[cloner_path.rebase(path, source, target)] = len(bodies)

    if simulation:
        global_sites, source_sites, root_sites = NewtonManager._cl_inject_sites(builder, source_builders)
    else:
        # Clear imported filters before merging into a fresh, compact final filter store.
        for imported in source_builders.values():
            imported.shape_collision_filter_pairs = []
            imported.shape_collision_group[:] = [0] * imported.shape_count
        global_sites, source_sites, root_sites = {}, {}, {}

    def record_source_particle_ranges(
        source: str,
        particle_offset: int,
        source_builder: ModelBuilder,
        source_xform: Sequence[float],
    ) -> None:
        record_registered_mpm_particle_ranges(
            source_import_results[source].get("path_particle_map", {}),
            particle_offset,
            builder=builder,
            source_builder=source_builder,
            source_xform=source_xform,
        )

    local_site_map, world_xforms, fabric_body_bindings = replicate_builder_mapping(
        builder=builder,
        plan=plan,
        instances=instances,
        positions=positions,
        quaternions=quaternions,
        source_builders=source_builders,
        env_ids=env_ids,
        reference_instances=reference_instances,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=NewtonManager._per_world_builder_hooks if simulation else (),
        source_builder_added=record_source_particle_ranges
        if simulation and NewtonManager._mpm_object_registry
        else None,
    )
    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    NewtonManager._cable_bindings = {}
    if cable_counts:
        shape_ids = {label: index for index, label in enumerate(builder.shape_label)}
        NewtonManager._cable_bindings = {
            path: [shape_ids[f"{path}_edge_capsule_{segment}"] for segment in range(count)]
            for path, count in cable_counts.items()
        }
    if simulation:
        NewtonManager._cl_site_index_map = site_index_map
        NewtonManager._cl_fabric_body_bindings = fabric_body_bindings
        NewtonManager._world_xforms = world_xforms
        NewtonManager._cl_protos = source_builders
        NewtonManager.set_builder(builder)
        NewtonManager._num_envs = len(env_ids)
    else:
        geometry_offsets = add_shadow_deformables_to_builder(
            builder, expand_deformable_entries(entries, instances, env_ids, positions)
        )
        backend_cfg = NewtonBackendCfg(builder=builder, device=sim.device, num_envs=len(env_ids), simulation=False)
        sim.physics_manager.register_callback(
            partial(NewtonManager._initialize_visualization_model, backend_cfg, geometry_offsets),
            PhysicsEvent.PHYSICS_READY,
            name="newton_visualization_model",
            wrap_weak_ref=False,
        )
    return builder, stage_info, site_index_map


class NewtonReplicateContext:
    """Build one Newton model from the sources routed to it in a clone plan."""

    replicate_priority = 0

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        """Initialize the context from its owning simulation."""
        self._sim = sim_context
        self.up_axis = up_axis

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> tuple[ModelBuilder, object, dict]:
        """Build and publish a Newton model from this context's source declarations."""
        instances = cloner_path.get_instance_paths(plan)
        return _replicate_newton(
            self._sim.stage,
            np.arange(len(plan.topology.world_prototype_layout)),
            self._sim,
            plan=plan,
            instances=tuple(instance for instance in instances if instance[0] in asset_prototype_ids),
            reference_instances=instances,
            positions=plan.positions,
            global_paths=cloner_path.get_shared_paths(instances),
            exclude_paths=tuple(
                source
                for index, source, _, world_ids in instances
                if index not in asset_prototype_ids and len(world_ids) and world_ids[0] != -1
            ),
            up_axis=self.up_axis,
        )


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
    world_masks, selected = np.unique(mapping.T, axis=0, return_inverse=True)
    members = [np.flatnonzero(mask) for mask in world_masks]
    shared = np.arange(len(sources), len(sources) + len(global_paths))
    prefix, suffix = destinations[0].split("{}", 1) if destinations else ("/World/envs/env_", "")
    plan = ClonePlan(
        PrototypeWorldTopology(
            len(sources) + len(global_paths),
            np.concatenate((shared, *members)).astype(np.int32),
            np.r_[0, len(shared), len(shared) + np.cumsum([len(world) for world in members])],
            selected.astype(np.int32),
        ),
        asset_cfgs=(*sources, *global_paths),
        env_template=prefix + "{}" + suffix.split("/", 1)[0],
        positions=positions,
    )
    instances = tuple(
        (index, source, destination, np.flatnonzero(mapping[index]))
        for index, (source, destination) in enumerate(zip(sources, destinations, strict=True))
    ) + tuple((int(index), path, path, np.array([-1])) for index, path in zip(shared, global_paths, strict=True))
    builder, stage_info, _ = _replicate_newton(
        stage,
        env_ids,
        PhysicsManager._sim,
        plan=plan,
        instances=instances,
        reference_instances=instances,
        positions=positions,
        global_paths=global_paths,
        up_axis=up_axis,
        quaternions=quaternions,
    )
    return builder, stage_info
