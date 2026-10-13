# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deformable prototype geometry and clone-plan expansion."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom

from .. import sim as sim_utils
from ..cloner import ClonePlan
from ..cloner import path as cloner_path
from ..sim.utils.queries import has_deformable_body_api
from .deformable_vis_remap import build_volume_vis_barycentric_remap
from .scene_data_backend import SceneDataFormat

logger = logging.getLogger(__name__)


@dataclass
class DeformableStageEntry:
    """Geometry and parent pose of one deformable prototype or instance."""

    root_path: str
    sim_mesh_path: str
    vis_mesh_path: str
    deformable_type: str
    vertex_count: int
    vis_vertex_count: int
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    vis_vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    vis_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Parent-frame world position [m]."""
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Parent-frame world orientation as an xyzw quaternion."""


def deformable_geometry_batches(
    entries: Sequence[DeformableStageEntry], offsets: Sequence[int], *, device: str
) -> list[tuple[SceneDataFormat.Points | SceneDataFormat.WeightedPoints, dict[str, tuple[int, int]]]]:
    """Bind visual mesh paths to native nodal ranges or static barycentric interpolation tables.

    Args:
        entries: Declared deformable instances in native body order.
        offsets: Native point offset for each entry.
        device: Device for static interpolation tables; native points are bound by the producer.

    Returns:
        Native-format publications and exact visual mesh paths with their output ranges.
        Interpolation tables are computed once per shared prototype, without packing native points.
    """
    direct, weighted = {}, {}
    indices, weights = [], []
    prototype_remaps = {}
    output_offset = 0
    for entry, offset in zip(entries, offsets, strict=True):
        key = (id(entry.vertices), id(entry.indices), id(entry.vis_vertices))
        if key not in prototype_remaps:
            if entry.vertices is entry.vis_vertices or np.array_equal(entry.vertices, entry.vis_vertices):
                prototype_remaps[key] = None
            else:
                if entry.deformable_type != "volume":
                    raise ValueError(f"Surface visual topology differs from its native nodes: {entry.root_path}")
                remap = build_volume_vis_barycentric_remap(entry.vertices, entry.indices, entry.vis_vertices)
                if remap is None:
                    raise ValueError(f"Cannot bind visual vertices to deformable tetrahedra: {entry.root_path}")
                prototype_remaps[key] = (remap.tet_vertex_indices.numpy(), remap.bary_weights.numpy())
        if prototype_remaps[key] is None:
            direct[entry.vis_mesh_path] = (int(offset), entry.vis_vertex_count)
            continue
        prototype_indices, prototype_weights = prototype_remaps[key]
        indices.append(prototype_indices + int(offset))
        weights.append(prototype_weights)
        weighted[entry.vis_mesh_path] = (output_offset, entry.vis_vertex_count)
        output_offset += entry.vis_vertex_count
    batches = [(SceneDataFormat.Points(), direct)] if direct else []
    if weighted:
        publication = SceneDataFormat.WeightedPoints()
        publication.indices = wp.array(np.concatenate(indices), dtype=wp.int32, device=device)
        publication.weights = wp.array(np.concatenate(weights), dtype=wp.float32, device=device)
        batches.append((publication, weighted))
    return batches


def _select_visual_mesh(vis_candidates: list, sim_mesh_prim, sim_vertex_count: int):
    """Choose the visual mesh among candidates when several non-sim meshes exist.

    Prefers a direct sibling of the simulation mesh, then name hints
    (``visual`` / ``render`` / ``display`` / ``proxy``), then a matching point
    count, then a shorter / lexicographically stable path.
    """
    if not vis_candidates:
        return sim_mesh_prim

    if len(vis_candidates) == 1:
        return vis_candidates[0]

    # Rare case that the sim mesh has multiple visual candidates.
    sim_parent = sim_mesh_prim.GetParent()

    def _score(prim) -> tuple:
        """Rank visual-mesh candidates; higher tuples are preferred by ``max``."""
        name = prim.GetName().lower()
        name_bonus = int(any(token in name for token in ("visual", "render", "display", "proxy")))
        sibling_bonus = int(prim.GetParent() == sim_parent)
        count_bonus = int(len(UsdGeom.PointBased(prim).GetPointsAttr().Get() or []) == sim_vertex_count)
        path = prim.GetPath().pathString
        return (sibling_bonus, name_bonus, count_bonus, -path.count("/"), path)

    return max(vis_candidates, key=_score)


def deformable_entry(root_prim: Usd.Prim) -> DeformableStageEntry | None:
    """Read simulation and visual geometry from one declared deformable prototype.

    Args:
        root_prim: Prim that carries a deformable-body API schema.

    Returns:
        Geometry with vertices [m] baked into the parent frame and its world pose [m, xyzw],
        or ``None`` when the prototype has no simulation mesh.
    """
    stage = root_prim.GetStage()
    root_path = root_prim.GetPath()
    mesh_prims = sim_utils.get_all_matching_child_prims(
        prim_path=root_path, predicate=lambda p: p.GetTypeName() in ("TetMesh", "Mesh"), stage=stage
    )
    tet_prims = [prim for prim in mesh_prims if prim.GetTypeName() == "TetMesh"]
    mesh_prims = [prim for prim in mesh_prims if prim.GetTypeName() == "Mesh"]
    sim_candidates, vis_candidates = [], []
    for prim in mesh_prims:
        schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
        is_sim = any("DeformableSimAPI" in schema for schema in schemas)
        (sim_candidates if is_sim else vis_candidates).append(prim)

    if tet_prims:
        if len(tet_prims) > 1:
            logger.warning(
                "Multiple TetMesh prims found under deformable root '%s'; using '%s' and ignoring %d others.",
                root_path,
                tet_prims[0].GetPath(),
                len(tet_prims) - 1,
            )

        deformable_type = "volume"
        sim_mesh_prim = tet_prims[0]
        tet_mesh = UsdGeom.TetMesh(sim_mesh_prim)
        pts = tet_mesh.GetPointsAttr().Get() or []
        raw_tet_indices = tet_mesh.GetTetVertexIndicesAttr().Get() or []
        indices = np.asarray(raw_tet_indices, dtype=np.int32).reshape(-1)
    elif mesh_prims:
        deformable_type = "surface"
        sim_mesh_prim = sim_candidates[0] if sim_candidates else mesh_prims[0]
        if not sim_candidates:
            vis_candidates = []
        usd_mesh = UsdGeom.Mesh(sim_mesh_prim)
        pts = usd_mesh.GetPointsAttr().Get() or []
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)
    else:
        logger.warning("Skipping deformable prim '%s': no simulation mesh found.", root_path)
        return None

    vis_mesh_prim = _select_visual_mesh(vis_candidates, sim_mesh_prim, len(pts))
    xform_cache = UsdGeom.XformCache()
    parent_transform = xform_cache.GetLocalToWorldTransform(root_prim.GetParent())
    world_to_parent = parent_transform.GetInverse()
    mesh_to_parent = np.asarray(xform_cache.GetLocalToWorldTransform(sim_mesh_prim) * world_to_parent)
    # USD affine transforms multiply row vectors.
    vertices = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    vertices = (vertices @ mesh_to_parent[:3, :3] + mesh_to_parent[3, :3]).astype(np.float32)
    vis_vertices = vertices
    if vis_mesh_prim != sim_mesh_prim:
        vis_pts = UsdGeom.PointBased(vis_mesh_prim).GetPointsAttr().Get()
        vis_to_parent = np.asarray(xform_cache.GetLocalToWorldTransform(vis_mesh_prim) * world_to_parent)
        vis_vertices = np.asarray(vis_pts or [], dtype=np.float32).reshape(-1, 3)
        vis_vertices = (vis_vertices @ vis_to_parent[:3, :3] + vis_to_parent[3, :3]).astype(np.float32)

    vis_indices = np.empty(0, dtype=np.int32)
    if vis_mesh_prim.GetTypeName() == "Mesh":
        vis_indices = np.asarray(UsdGeom.Mesh(vis_mesh_prim).GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)

    rotation = parent_transform.ExtractRotationQuat()
    return DeformableStageEntry(
        root_path=str(root_path),
        sim_mesh_path=str(sim_mesh_prim.GetPath()),
        vis_mesh_path=str(vis_mesh_prim.GetPath()),
        deformable_type=deformable_type,
        vertex_count=len(pts),
        vis_vertex_count=len(vis_vertices),
        vertices=vertices,
        indices=indices,
        vis_vertices=vis_vertices,
        vis_indices=vis_indices,
        init_pos=tuple(parent_transform.ExtractTranslation()),
        init_rot=(*rotation.GetImaginary(), rotation.GetReal()),
    )


def deformable_prototypes(
    stage: Usd.Stage, plan: ClonePlan, *, exclude_paths: Sequence[str] = ()
) -> list[DeformableStageEntry]:
    """Read deformable geometry beneath the prototypes imported by one backend.

    Args:
        stage: Stage containing the authored asset prototypes.
        plan: Declared asset prototypes and their world topology.
        exclude_paths: Prototypes routed to other contexts, excluded even beneath shared roots.

    Returns:
        Prototype geometry owned by the caller, including shared assets once.
    """
    authored = cloner_path.get_asset_prototype_paths(plan)
    templates, starts = cloner_path.get_world_prototype_asset_templates(plan)
    prototype_ids = np.unique(plan.topology.world_prototypes[starts[1] :])
    source_paths = [authored[i] for i in prototype_ids if authored[i] is not None and authored[i] not in exclude_paths]
    destination_paths = templates[starts[1] :]
    selected_sources = {Sdf.Path(source) for source in source_paths}
    sources = selected_sources | {Sdf.Path(source) for source in exclude_paths}
    entries = []
    for root in Sdf.Path.RemoveDescendentPaths([*source_paths, *templates[: starts[1]]]):
        prims = iter(Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()))
        for prim in prims:
            path = prim.GetPath()
            owner = path
            while owner != Sdf.Path.absoluteRootPath and owner not in sources:
                owner = owner.GetParentPath()
            if owner in sources and owner not in selected_sources:
                if not any(source.HasPrefix(path) for source in selected_sources):
                    prims.PruneChildren()
                continue
            # Shared roots may contain a replicated namespace: never inspect its generated clones.
            if owner not in sources and any(cloner_path.match(str(path), template) for template in destination_paths):
                if not any(source.HasPrefix(path) for source in selected_sources):
                    prims.PruneChildren()
                    continue
            if prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.BasisCurves) or not has_deformable_body_api(prim):
                continue
            if (entry := deformable_entry(prim)) is not None:
                entries.append(entry)
    return entries


def expand_deformable_entries(
    prototypes: Sequence[DeformableStageEntry],
    plan: ClonePlan,
    env_ids: np.ndarray,
    positions: np.ndarray | None = None,
    *,
    imported_sources: Collection[str] | None = None,
) -> list[DeformableStageEntry]:
    """Expand backend-owned prototype geometry without copying its vertex arrays or reading USD.

    Args:
        prototypes: Geometry captured during this backend's prototype import.
        plan: Declared asset prototypes and their world topology.
        env_ids: Target environment ids.
        positions: Environment origins [m], shape [num_envs, 3].
        imported_sources: Source builders owned by this backend. When omitted, use all plan sources.

    Returns:
        Destination geometry records, including shared geometry once. Parent-frame world poses [m, xyzw]
        include the clone translation; topology and vertex arrays remain shared with the prototype.
    """
    entries: dict[str, tuple[str, DeformableStageEntry]] = {}
    source_instances = defaultdict(list)
    source_worlds = {str(env_id): world for world, env_id in enumerate(env_ids)}
    sources = cloner_path.get_asset_prototype_paths(plan)
    templates, starts, world_ids, world_starts = cloner_path.get_world_prototype_asset_templates(
        plan, include_world_indices=True
    )
    for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
        columns = world_ids[world_starts[group] : world_starts[group + 1]]
        for index in range(*starts[group : group + 2]):
            source = sources[plan.topology.world_prototypes[index]]
            if imported_sources is not None and source not in imported_sources:
                continue
            source_instances[Sdf.Path(source)].append((source, templates[index], columns))
    for entry in prototypes:
        # An imported parent and child can own this prototype in different world compositions.
        owners = [path for path in Sdf.Path(entry.root_path).GetPrefixes() if path in source_instances]
        if not owners:
            entries[entry.root_path] = (entry.root_path, entry)
            continue
        for owner in owners:
            source = source_instances[owner][0][0]
            match = cloner_path.match(source, plan.env_template)
            # Match the authored source world; remapped IDs retain the first destination convention.
            source_world = source_worlds.get(match.instance, source_instances[owner][0][2][0]) if match else None
            for source, template, columns in source_instances[owner]:
                for column in columns:
                    target = template.format(int(env_ids[column]))
                    offset = (
                        0
                        if positions is None
                        else positions[column] - (positions[source_world] if source_world is not None else 0)
                    )
                    cloned = replace(
                        entry,
                        root_path=cloner_path.rebase(entry.root_path, source, target),
                        sim_mesh_path=cloner_path.rebase(entry.sim_mesh_path, source, target),
                        vis_mesh_path=cloner_path.rebase(entry.vis_mesh_path, source, target),
                        init_pos=tuple(np.asarray(entry.init_pos) + offset),
                    )
                    # An overlapping child declaration wins within the same destination.
                    if cloned.root_path not in entries or len(target) > len(entries[cloned.root_path][0]):
                        entries[cloned.root_path] = (target, cloned)
    return [entry for _, entry in entries.values()]
