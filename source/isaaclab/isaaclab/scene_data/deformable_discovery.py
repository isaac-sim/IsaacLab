# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery helpers for PhysX/OVPhysX deformable geometry."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils

logger = logging.getLogger(__name__)


@dataclass
class DeformableStageEntry:
    """One deformable asset instance discovered on the USD stage."""

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
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


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
    return np.array([[float(point[0]), float(point[1]), float(point[2])] for point in points], dtype=np.float32)


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


def _classify_deformable_meshes(
    root_prim,
) -> tuple[str, object, object, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify the simulation and visual meshes under a deformable root prim.

    Args:
        root_prim: Prim that carries ``OmniPhysicsDeformableBodyAPI``.

    Returns:
        A 9-tuple:
            deformable_type: ``"volume"`` for a tet sim mesh, else ``"surface"``.
            sim_mesh_prim: Simulation mesh prim (``TetMesh`` or ``Mesh``).
            vis_mesh_prim: Selected visual mesh prim (may equal the sim mesh).
            vertex_count: Number of simulation mesh vertices.
            vis_vertex_count: Number of visual mesh vertices.
            vertices: Sim vertices baked into the deformable parent frame [m].
            indices: Flattened sim topology (4 ints per tet, or face-vertex indices).
            vis_vertices: Visual vertices baked into the same parent frame [m].
            vis_indices: Flattened visual triangle indices (empty when not a ``Mesh``).

    Raises:
        ValueError: If no simulation mesh is found under ``root_prim``.
    """
    stage = root_prim.GetStage()
    root_path = root_prim.GetPath()
    tet_prims = sim_utils.get_all_matching_child_prims(
        prim_path=root_path, predicate=lambda p: p.GetTypeName() == "TetMesh", stage=stage
    )
    mesh_prims = sim_utils.get_all_matching_child_prims(
        prim_path=root_path, predicate=lambda p: p.GetTypeName() == "Mesh", stage=stage
    )

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
        raise ValueError(f"No simulation mesh found under deformable root '{root_path}'.")

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

    return (
        deformable_type,
        sim_mesh_prim,
        vis_mesh_prim,
        len(pts),
        vis_count,
        vertices,
        indices,
        vis_vertices,
        vis_indices,
    )


def discover_deformables_on_stage(stage: Usd.Stage) -> list[DeformableStageEntry]:
    """Discover PhysX/OVPhysX deformable bodies under ``stage``.

    Callers that need the result in more than one place should keep the returned
    list and pass it explicitly (for example via ``entries=``) rather than relying
    on a process-global stage cache.

    Args:
        stage: USD stage to traverse.

    Returns:
        One :class:`DeformableStageEntry` per prim with ``OmniPhysicsDeformableBodyAPI``.
    """
    entries: list[DeformableStageEntry] = []
    for prim in stage.Traverse():
        if not _prim_has_schema(prim, "OmniPhysicsDeformableBodyAPI"):
            continue

        try:
            (
                deformable_type,
                sim_mesh_prim,
                vis_mesh_prim,
                vertex_count,
                vis_vertex_count,
                vertices,
                indices,
                vis_vertices,
                vis_indices,
            ) = _classify_deformable_meshes(prim)
        except ValueError as exc:
            logger.warning("Skipping deformable prim '%s': %s", prim.GetPath(), exc)
            continue

        entries.append(
            DeformableStageEntry(
                root_path=prim.GetPath().pathString,
                sim_mesh_path=sim_mesh_prim.GetPath().pathString,
                vis_mesh_path=vis_mesh_prim.GetPath().pathString,
                deformable_type=deformable_type,
                vertex_count=vertex_count,
                vis_vertex_count=vis_vertex_count,
                vertices=vertices,
                indices=indices,
                vis_vertices=vis_vertices,
                vis_indices=vis_indices,
            )
        )
    return entries


def group_deformable_root_paths_for_views(
    root_paths: list[str],
    path_to_type: dict[str, str],
) -> dict[str, tuple[list[str], list[str]]]:
    """Group deformable root paths into PhysX/OVPhysX view patterns and exact paths.

    Replicated env assets with the same trailing prim name collapse to
    ``/World/envs/env_*`` wildcard patterns; singleton paths stay exact.

    Args:
        root_paths: Discovered deformable root prim paths.
        path_to_type: Map from root path to ``volume`` or ``surface``.

    Returns:
        Dict keyed by deformable type with ``(wildcard_patterns, exact_paths)`` lists.
        Wildcard patterns are sorted; exact paths preserve discovery order within type.
    """
    non_rigid_names: set[str] = set()
    for path in root_paths:
        if re.search(r"/World/envs/env_\d+/", path):
            non_rigid_names.add(path.rsplit("/", 1)[-1])

    typed_patterns: dict[str, set[str]] = {"volume": set(), "surface": set()}
    typed_exact: dict[str, list[str]] = {"volume": [], "surface": []}
    for path in root_paths:
        body_name = path.rsplit("/", 1)[-1]
        wildcard = path_to_env_wildcard(path)
        deformable_type = path_to_type[path]
        if body_name in non_rigid_names and wildcard != path:
            typed_patterns[deformable_type].add(wildcard)
        else:
            typed_exact[deformable_type].append(path)

    return {
        deformable_type: ([*sorted(typed_patterns[deformable_type])], typed_exact[deformable_type])
        for deformable_type in ("volume", "surface")
    }


def sort_deformable_entries_for_geometry_sync(entries: list[DeformableStageEntry]) -> list[DeformableStageEntry]:
    """Return deformable entries in SceneData geometry path order (volume, then surface)."""
    type_rank = {"volume": 0, "surface": 1}
    return sorted(entries, key=lambda entry: (type_rank.get(entry.deformable_type, 2), entry.root_path))


def path_to_env_wildcard(path: str) -> str:
    """Rewrite ``env_<id>`` segments to ``env_*`` for PhysX tensor patterns."""
    return re.sub(r"/World/envs/env_\d+", "/World/envs/env_*", path)


def path_to_env_regex(path: str) -> str:
    """Rewrite ``env_<id>`` segments to ``env_.*`` for Isaac Lab asset regex paths."""
    return re.sub(r"/World/envs/env_\d+", "/World/envs/env_.*", path)


def build_deformable_vertex_count_lookup(entries: list[DeformableStageEntry]) -> dict[str, int]:
    """Map deformable root, simulation-mesh, and visual-mesh paths to unpadded vertex counts."""
    path_to_count: dict[str, int] = {}
    for entry in entries:
        path_to_count[entry.root_path] = entry.vertex_count
        path_to_count[entry.sim_mesh_path] = entry.vertex_count
        path_to_count[entry.vis_mesh_path] = entry.vis_vertex_count
    return path_to_count


def build_deformable_root_path_lookup(entries: list[DeformableStageEntry]) -> dict[str, str]:
    """Map deformable root, simulation-mesh, and visual-mesh paths to discovered roots."""
    path_to_root: dict[str, str] = {}
    for entry in entries:
        path_to_root[entry.root_path] = entry.root_path
        path_to_root[entry.sim_mesh_path] = entry.root_path
        path_to_root[entry.vis_mesh_path] = entry.root_path
    return path_to_root


def _env_relative_suffix(path: str) -> str | None:
    """Return the path suffix after ``/World/envs/env_<id>/``, or ``None`` if absent."""
    match = re.search(r"/World/envs/env_\d+/(.*)", path)
    return match.group(1) if match else None


def _rewrite_env_prefix(template_path: str, source_path: str) -> str:
    """Copy the ``env_<id>`` segment from ``source_path`` into ``template_path``."""
    match = re.search(r"/World/envs/(env_\d+)", source_path)
    if match is None:
        return template_path
    return re.sub(r"/World/envs/env_\d+", f"/World/envs/{match.group(1)}", template_path)


def resolve_deformable_vertex_count(path: str, path_to_count: dict[str, int], *, fallback: int) -> int:
    """Resolve an unpadded vertex count for a deformable-related prim path.

    Tries the exact path, ancestor/descendant relationships, and env-relative
    suffixes (views may report a different child under the same env asset).
    Falls back only when no discovered count matches.

    Args:
        path: Deformable root or mesh prim path reported by a physics view.
        path_to_count: Lookup from :func:`build_deformable_vertex_count_lookup`.
        fallback: Value returned when no discovered count matches ``path``.

    Returns:
        Unpadded vertex count for ``path``, or ``fallback``.
    """
    if path in path_to_count:
        return int(path_to_count[path])

    normalized = path.rstrip("/")
    parts = normalized.split("/")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:end])
        if candidate in path_to_count:
            return int(path_to_count[candidate])

    for key, count in path_to_count.items():
        key_normalized = key.rstrip("/")
        if key_normalized.startswith(normalized + "/") or normalized.startswith(key_normalized + "/"):
            return int(count)

    suffix = _env_relative_suffix(normalized)
    if suffix is not None:
        for key, count in path_to_count.items():
            key_suffix = _env_relative_suffix(key)
            if key_suffix is None:
                continue
            if suffix == key_suffix or suffix.startswith(key_suffix + "/") or key_suffix.startswith(suffix + "/"):
                return int(count)

    return int(fallback)


def resolve_deformable_root_path(path: str, path_to_root: dict[str, str], *, fallback: str | None = None) -> str:
    """Resolve a view-reported deformable path to its discovered root prim path.

    Uses the same matching strategy as :func:`resolve_deformable_vertex_count`.
    Env-relative suffix matches rewrite the matched root's ``env_<id>`` to the
    reported path's environment so SceneData geometry mapping stays 1:1 with
    shadow entity ``root_path`` values.

    Args:
        path: Deformable root or mesh prim path reported by a physics view.
        path_to_root: Lookup from :func:`build_deformable_root_path_lookup`.
        fallback: Value returned when no discovered root matches ``path``.
            Defaults to ``path``.

    Returns:
        Discovered deformable root path, or ``fallback`` / ``path``.
    """
    if fallback is None:
        fallback = path

    if path in path_to_root:
        return path_to_root[path]

    normalized = path.rstrip("/")
    parts = normalized.split("/")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:end])
        if candidate in path_to_root:
            return path_to_root[candidate]

    for key, root in path_to_root.items():
        key_normalized = key.rstrip("/")
        if key_normalized.startswith(normalized + "/") or normalized.startswith(key_normalized + "/"):
            return root

    suffix = _env_relative_suffix(normalized)
    if suffix is not None:
        for key, root in path_to_root.items():
            key_suffix = _env_relative_suffix(key)
            if key_suffix is None:
                continue
            if suffix == key_suffix or suffix.startswith(key_suffix + "/") or key_suffix.startswith(suffix + "/"):
                return _rewrite_env_prefix(root, normalized)

    return fallback
