# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deformable prototype geometry and clone-plan expansion."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom

from .. import sim as sim_utils
from ..cloner.path import match, rebase
from ..sim.utils.queries import has_deformable_body_api
from .deformable_vis_remap import build_volume_vis_barycentric_remap
from .scene_data_backend import SceneDataFormat

if TYPE_CHECKING:
    from ..cloner.clone_plan import ClonePlan

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
    entries: Sequence[DeformableStageEntry], points: wp.array, offsets: Sequence[int]
) -> list[tuple[SceneDataFormat.Points | SceneDataFormat.WeightedPoints, dict[str, tuple[int, int]]]]:
    """Bind visual mesh paths to native nodal ranges or static barycentric interpolation tables.

    Args:
        entries: Declared deformable instances in native body order.
        points: Flat native nodal positions [m], including any body padding.
        offsets: Native point offset for each entry.

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
            if entry.vertex_count == entry.vis_vertex_count and np.array_equal(entry.vertices, entry.vis_vertices):
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
    batches = []
    if direct:
        publication = SceneDataFormat.Points()
        publication.points = points
        batches.append((publication, direct))
    if weighted:
        publication = SceneDataFormat.WeightedPoints()
        publication.points = points
        publication.indices = wp.array(np.concatenate(indices), dtype=wp.int32, device=points.device)
        publication.weights = wp.array(np.concatenate(weights), dtype=wp.float32, device=points.device)
        batches.append((publication, weighted))
    return batches


def _matrix4d_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    """Convert a USD matrix to a host ``(4, 4)`` float64 array."""
    return np.array([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a USD ``(4, 4)`` transform to ``(N, 3)`` points [m], returning float32.

    USD ``Gf.Matrix4d.Transform`` uses row-vector convention (``p @ M``). The numpy
    matrix from :func:`_matrix4d_to_numpy` stores ``matrix[i, j] = usd[i][j]``, so the
    matching host multiply is ``hom @ matrix``, not ``matrix @ hom``.
    """
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([points.astype(np.float64, copy=False), ones], axis=1)
    baked = (hom @ matrix)[:, :3]
    return baked.astype(np.float32, copy=False)


def _usd_points_to_numpy(points) -> np.ndarray:
    """Convert USD point arrays to ``(N, 3)`` float32."""
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _get_applied_schema_names(prim) -> set[str]:
    """Return applied API schema names from composed schemas and explicit ``apiSchemas`` metadata."""
    names = set(prim.GetAppliedSchemas())
    api_schemas = prim.GetMetadata("apiSchemas")
    if isinstance(api_schemas, Sdf.TokenListOp):
        names.update(str(token) for token in api_schemas.explicitItems)
    return names


def _prim_has_schema(prim, schema_substring: str) -> bool:
    """Return ``True`` if any applied API schema name contains ``schema_substring``."""
    return any(schema_substring in name for name in _get_applied_schema_names(prim))


def _mesh_point_count(prim) -> int:
    """Return the number of points authored on a Mesh or TetMesh prim."""
    if prim.GetTypeName() not in ("Mesh", "TetMesh"):
        return 0
    pts = UsdGeom.PointBased(prim).GetPointsAttr().Get()
    return len(pts or [])


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
    sim_parent_path = sim_parent.GetPath() if sim_parent is not None and sim_parent.IsValid() else None

    def _score(prim) -> tuple:
        """Rank visual-mesh candidates; higher tuples are preferred by ``max``."""
        name = prim.GetName().lower()
        name_bonus = int(any(token in name for token in ("visual", "render", "display", "proxy")))
        sibling_bonus = int(
            sim_parent_path is not None
            and prim.GetParent() is not None
            and prim.GetParent().GetPath() == sim_parent_path
        )
        count_bonus = int(_mesh_point_count(prim) == sim_vertex_count)
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
        vis_candidates = [p for p in mesh_prims if not _prim_has_schema(p, "DeformableSimAPI")]
        tet_mesh = UsdGeom.TetMesh(sim_mesh_prim)
        pts = tet_mesh.GetPointsAttr().Get() or []
        raw_tet_indices = tet_mesh.GetTetVertexIndicesAttr().Get() or []
        indices = np.array([int(v) for vec4i in raw_tet_indices for v in vec4i], dtype=np.int32)
    elif mesh_prims:
        deformable_type = "surface"
        sim_candidates: list = []
        vis_candidates: list = []
        for prim in mesh_prims:
            if _prim_has_schema(prim, "DeformableSimAPI"):
                sim_candidates.append(prim)
            else:
                vis_candidates.append(prim)
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
    vis_pts = (
        UsdGeom.Mesh(vis_mesh_prim).GetPointsAttr().Get()
        if vis_mesh_prim.GetTypeName() == "Mesh"
        else UsdGeom.TetMesh(vis_mesh_prim).GetPointsAttr().Get()
    )
    vis_count = len(vis_pts or [])

    xform_cache = UsdGeom.XformCache()
    mesh_to_parent_frame = _matrix4d_to_numpy(
        xform_cache.GetLocalToWorldTransform(sim_mesh_prim)
        * xform_cache.GetLocalToWorldTransform(root_prim.GetParent()).GetInverse()
    )
    vertices = _transform_points(mesh_to_parent_frame, _usd_points_to_numpy(pts))

    vis_mesh_to_parent_frame = _matrix4d_to_numpy(
        xform_cache.GetLocalToWorldTransform(vis_mesh_prim)
        * xform_cache.GetLocalToWorldTransform(root_prim.GetParent()).GetInverse()
    )
    vis_vertices = _transform_points(vis_mesh_to_parent_frame, _usd_points_to_numpy(vis_pts or []))

    vis_indices = np.empty(0, dtype=np.int32)
    if vis_mesh_prim.GetTypeName() == "Mesh":
        vis_indices = np.asarray(UsdGeom.Mesh(vis_mesh_prim).GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)

    parent_transform = xform_cache.GetLocalToWorldTransform(root_prim.GetParent())
    rotation = parent_transform.ExtractRotationQuat()
    return DeformableStageEntry(
        root_path=str(root_path),
        sim_mesh_path=str(sim_mesh_prim.GetPath()),
        vis_mesh_path=str(vis_mesh_prim.GetPath()),
        deformable_type=deformable_type,
        vertex_count=len(pts),
        vis_vertex_count=vis_count,
        vertices=vertices,
        indices=indices,
        vis_vertices=vis_vertices,
        vis_indices=vis_indices,
        init_pos=tuple(parent_transform.ExtractTranslation()),
        init_rot=(*rotation.GetImaginary(), rotation.GetReal()),
    )


def deformable_prototypes(
    stage: Usd.Stage, plan: ClonePlan, rows: Sequence[int] | None = None
) -> list[DeformableStageEntry]:
    """Read deformable geometry beneath the prototypes imported by one backend.

    Args:
        stage: Stage containing the authored asset prototypes.
        plan: Generic replication layout; it is not modified.
        rows: Rows imported by this backend; ``None`` selects all active rows.

    Returns:
        Prototype geometry owned by the caller, including shared assets once.
    """
    selected = set(range(len(plan.sources)) if rows is None else rows)
    selected.intersection_update(np.flatnonzero(plan.clone_mask.any(axis=1)))
    sources = {Sdf.Path(source) for source in plan.sources}
    selected_sources = {Sdf.Path(plan.sources[row]) for row in selected}
    roots = Sdf.Path.RemoveDescendentPaths([plan.sources[row] for row in selected] + list(plan.global_paths))
    entries = []
    for root in roots:
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
            if owner not in sources and any(match(str(path), template) for template in plan.destinations):
                if not any(source.HasPrefix(path) for source in selected_sources):
                    prims.PruneChildren()
                    continue
            if prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.BasisCurves) or not has_deformable_body_api(prim):
                continue
            if (entry := deformable_entry(prim)) is not None:
                entries.append(entry)
    return entries


def expand_deformable_entries(
    plan: ClonePlan, prototypes: Sequence[DeformableStageEntry], rows: Sequence[int] | None = None
) -> list[DeformableStageEntry]:
    """Expand backend-owned prototype geometry without copying its vertex arrays or reading USD.

    Args:
        plan: Generic source-to-destination mapping.
        prototypes: Geometry captured during this backend's prototype import.
        rows: Source rows consumed by this backend; ``None`` selects all rows.

    Returns:
        Destination geometry records, including shared geometry once. Parent-frame world poses [m, xyzw]
        include the clone translation; topology and vertex arrays remain shared with the prototype.
    """
    entries: dict[str, tuple[str, DeformableStageEntry]] = {}
    selected = set(range(len(plan.sources)) if rows is None else rows)
    source_rows = defaultdict(list)
    for row, source in enumerate(plan.sources):
        source_rows[Sdf.Path(source)].append(row)
    for entry in prototypes:
        owner = Sdf.Path(entry.root_path)
        while owner != Sdf.Path.absoluteRootPath and owner not in source_rows:
            owner = owner.GetParentPath()
        if owner not in source_rows:
            entries[entry.root_path] = (entry.root_path, entry)
            continue
        for row in source_rows[owner]:
            if row not in selected:
                continue
            columns = np.flatnonzero(plan.clone_mask[row])
            for column in columns:
                target = plan.destinations[row].format(int(plan.env_ids[column]))
                offset = 0 if plan.positions is None else plan.positions[column] - plan.positions[columns[0]]
                cloned = replace(
                    entry,
                    root_path=rebase(entry.root_path, plan.sources[row], target),
                    sim_mesh_path=rebase(entry.sim_mesh_path, plan.sources[row], target),
                    vis_mesh_path=rebase(entry.vis_mesh_path, plan.sources[row], target),
                    init_pos=tuple(np.asarray(entry.init_pos) + offset),
                )
                if cloned.root_path not in entries or len(target) > len(entries[cloned.root_path][0]):
                    entries[cloned.root_path] = (target, cloned)
    return [entry for _, entry in entries.values()]
