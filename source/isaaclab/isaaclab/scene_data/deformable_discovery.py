# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deformable prototype geometry and clone-plan expansion."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from .. import sim as sim_utils
from ..cloner.path import rebase

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


def discover_deformables_on_stage(
    stage: Usd.Stage, *, root_paths: Sequence[str] | None = None
) -> list[DeformableStageEntry]:
    """Discover PhysX/OVPhysX deformable bodies under ``stage``.

    Callers that need the result in more than one place should keep the returned
    list and pass it explicitly (for example via ``entries=``) rather than relying
    on a process-global stage cache.

    Args:
        stage: USD stage to traverse.
        root_paths: Declared asset roots to inspect. None selects the complete stage.

    Returns:
        One :class:`DeformableStageEntry` per prim with ``OmniPhysicsDeformableBodyAPI``.
    """
    entries: list[DeformableStageEntry] = []
    prims = (
        stage.Traverse()
        if root_paths is None
        else chain.from_iterable(
            Usd.PrimRange(stage.GetPrimAtPath(path)) for path in Sdf.Path.RemoveDescendentPaths(root_paths)
        )
    )
    for prim in prims:
        if not _prim_has_schema(prim, "OmniPhysicsDeformableBodyAPI"):
            continue

        if (entry := deformable_entry(prim)) is not None:
            entries.append(entry)
    return entries


def deformable_entries(plan: ClonePlan, rows: Sequence[int] | None = None) -> list[DeformableStageEntry]:
    """Expand plan-owned prototypes to their exact destination paths without copying vertex arrays.

    Args:
        plan: Clone plan whose geometry was compiled before replication.
        rows: Source rows consumed by this backend; ``None`` selects all rows.

    Returns:
        Destination geometry records, including shared geometry once. Parent-frame world poses [m, xyzw]
        include the clone translation; topology and vertex arrays remain shared with the prototype.
    """
    entries = {entry.root_path: entry for entry in plan.deformables[None]}
    for row in range(len(plan.sources)) if rows is None else rows:
        columns = np.flatnonzero(plan.clone_mask[row])
        for entry in plan.deformables[row]:
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
                entries.setdefault(cloned.root_path, cloned)
    return list(entries.values())


def sort_deformable_entries_for_geometry_sync(entries: list[DeformableStageEntry]) -> list[DeformableStageEntry]:
    """Return deformable entries in SceneData geometry path order (volume, then surface)."""
    type_rank = {"volume": 0, "surface": 1}
    return sorted(entries, key=lambda entry: (type_rank.get(entry.deformable_type, 2), entry.root_path))
