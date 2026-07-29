# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery helpers for PhysX/OVPhysX deformable geometry."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils

logger = logging.getLogger(__name__)


def _get_applied_schema_names(prim) -> set[str]:
    """Return applied API schema names from composed schemas and explicit ``apiSchemas`` metadata."""
    names = set(prim.GetAppliedSchemas())
    api_schemas = prim.GetMetadata("apiSchemas")
    if isinstance(api_schemas, Sdf.TokenListOp):
        names.update(str(token) for token in api_schemas.explicitItems)
    return names


def _prim_has_schema(prim, schema_substring: str) -> bool:
    return any(schema_substring in name for name in _get_applied_schema_names(prim))


@dataclass
class DeformableStageEntry:
    """One deformable asset instance discovered on the USD stage."""

    root_path: str
    sim_mesh_path: str
    vis_mesh_path: str
    deformable_type: str
    vertex_count: int
    vis_vertex_count: int
    vertices: list = field(default_factory=list)
    indices: list = field(default_factory=list)
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def _is_sim_mesh(prim) -> bool:
    return _prim_has_schema(prim, "DeformableSimAPI")


def _collect_type_prims(root_prim, type_name: str) -> list:
    """Collect prims of ``type_name`` under ``root_prim``, including sibling meshes.

    When ``OmniPhysicsDeformableBodyAPI`` is applied directly on a simulation TetMesh or
    Mesh, the visual Mesh is often a sibling under the parent Xform. Child-only search
    would miss that sibling and fall back to binding the sim mesh for OVRTX.

    Sibling search only inspects direct children of the parent so nested meshes under
    unrelated sibling branches are not treated as visual candidates.
    """
    stage = root_prim.GetStage()
    root_path = root_prim.GetPath()
    prims = list(sim_utils.get_all_matching_child_prims(root_path, lambda p: p.GetTypeName() == type_name, stage=stage))
    parent = root_prim.GetParent()
    if parent is not None and parent.IsValid() and root_prim.GetTypeName() in ("TetMesh", "Mesh"):
        known_paths = {prim.GetPath() for prim in prims}
        for child in parent.GetChildren():
            if child.GetTypeName() != type_name or child.GetPath() in known_paths:
                continue
            prims.append(child)
    return prims


def _mesh_point_count(prim) -> int:
    """Return the number of points authored on a Mesh or TetMesh prim."""
    if prim.GetTypeName() == "Mesh":
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    elif prim.GetTypeName() == "TetMesh":
        pts = UsdGeom.TetMesh(prim).GetPointsAttr().Get()
    else:
        return 0
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

    sim_parent = sim_mesh_prim.GetParent()
    sim_parent_path = sim_parent.GetPath() if sim_parent is not None and sim_parent.IsValid() else None

    def _score(prim) -> tuple:
        name = prim.GetName().lower()
        name_bonus = int(any(token in name for token in ("visual", "render", "display", "proxy")))
        sibling_bonus = int(
            sim_parent_path is not None and prim.GetParent() is not None and prim.GetParent().GetPath() == sim_parent_path
        )
        count_bonus = int(_mesh_point_count(prim) == sim_vertex_count)
        path = prim.GetPath().pathString
        return (sibling_bonus, name_bonus, count_bonus, -path.count("/"), path)

    return max(vis_candidates, key=_score)


def _classify_deformable_meshes(root_prim) -> tuple[str, object, object, int, int, list, list]:
    """Return deformable type, sim mesh prim, vis mesh prim, counts, vertices, and indices."""
    import warp as wp

    root_path = root_prim.GetPath()
    tet_prims = _collect_type_prims(root_prim, "TetMesh")
    mesh_prims = _collect_type_prims(root_prim, "Mesh")

    if len(tet_prims) == 1:
        mesh_prim = tet_prims[0]
        vis_candidates = [p for p in mesh_prims if not _is_sim_mesh(p)]
        deformable_type = "volume"
        tet_mesh = UsdGeom.TetMesh(mesh_prim)
        pts = tet_mesh.GetPointsAttr().Get() or []
        raw_tet_indices = tet_mesh.GetTetVertexIndicesAttr().Get() or []
        indices: list[int] = []
        for vec4i in raw_tet_indices:
            indices.extend([int(vec4i[0]), int(vec4i[1]), int(vec4i[2]), int(vec4i[3])])
    elif mesh_prims:
        deformable_type = "surface"
        sim_candidates = [p for p in mesh_prims if _is_sim_mesh(p)]
        vis_candidates = [p for p in mesh_prims if not _is_sim_mesh(p)]
        mesh_prim = sim_candidates[0] if sim_candidates else mesh_prims[0]
        if not sim_candidates:
            vis_candidates = []
        usd_mesh = UsdGeom.Mesh(mesh_prim)
        pts = usd_mesh.GetPointsAttr().Get() or []
        indices = list(usd_mesh.GetFaceVertexIndicesAttr().Get() or [])
    else:
        raise ValueError(f"No simulation mesh found under deformable root '{root_path}'.")

    vis_mesh_prim = _select_visual_mesh(vis_candidates, mesh_prim, len(pts))
    vis_pts = (
        UsdGeom.Mesh(vis_mesh_prim).GetPointsAttr().Get()
        if vis_mesh_prim.GetTypeName() == "Mesh"
        else UsdGeom.TetMesh(vis_mesh_prim).GetPointsAttr().Get()
    )
    vis_count = len(vis_pts or [])

    xform_cache = UsdGeom.XformCache()
    mesh_to_parent_frame = (
        xform_cache.GetLocalToWorldTransform(mesh_prim)
        * xform_cache.GetLocalToWorldTransform(root_prim.GetParent()).GetInverse()
    )

    vertices: list = []
    for point in pts:
        baked = mesh_to_parent_frame.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        vertices.append(wp.vec3(float(baked[0]), float(baked[1]), float(baked[2])))

    return deformable_type, mesh_prim, vis_mesh_prim, len(pts), vis_count, vertices, indices


def discover_deformables_on_stage(stage: Usd.Stage) -> list[DeformableStageEntry]:
    """Discover PhysX/OVPhysX deformable bodies under ``stage``.

    Returns:
        One :class:`DeformableStageEntry` per prim with ``OmniPhysicsDeformableBodyAPI``.
    """
    entries: list[DeformableStageEntry] = []
    for prim in stage.Traverse():
        if not _prim_has_schema(prim, "OmniPhysicsDeformableBodyAPI"):
            continue
        try:
            deformable_type, sim_mesh_prim, vis_mesh_prim, vertex_count, vis_vertex_count, vertices, indices = (
                _classify_deformable_meshes(prim)
            )
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


def _env_relative_suffix(path: str) -> str | None:
    match = re.search(r"/World/envs/env_\d+/(.*)", path)
    return match.group(1) if match else None


def resolve_deformable_vertex_count(path: str, path_to_count: dict[str, int], *, fallback: int) -> int:
    """Resolve an unpadded vertex count for a deformable-related prim path.

    Tries the exact path, ancestor/descendant relationships, and env-relative
    suffixes (views may report a different child under the same env asset).
    Falls back only when no discovered count matches.
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
