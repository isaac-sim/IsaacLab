# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery helpers for PhysX/OVPhysX deformable geometry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils


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
    return any("DeformableSimAPI" in api for api in prim.GetPrimTypeInfo().GetAppliedAPISchemas())


def _classify_deformable_meshes(root_prim) -> tuple[str, object, object, int, int, list, list]:
    """Return deformable type, sim mesh prim, vis mesh prim, counts, vertices, and indices."""
    import warp as wp

    root_path = root_prim.GetPath()
    tet_prims = sim_utils.get_all_matching_child_prims(root_path, lambda p: p.GetTypeName() == "TetMesh")
    mesh_prims = sim_utils.get_all_matching_child_prims(root_path, lambda p: p.GetTypeName() == "Mesh")

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

    vis_mesh_prim = vis_candidates[0] if vis_candidates else mesh_prim
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
        if "OmniPhysicsDeformableBodyAPI" not in prim.GetAppliedSchemas():
            continue
        try:
            deformable_type, sim_mesh_prim, vis_mesh_prim, vertex_count, vis_vertex_count, vertices, indices = (
                _classify_deformable_meshes(prim)
            )
        except ValueError:
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


def compact_env_wildcard_paths(paths: list[str]) -> tuple[list[str], list[int]]:
    """Compact env-expanded paths into wildcard patterns with per-path instance counts.

    Returns:
        Tuple of unique wildcard patterns and the number of expanded paths represented
        by each pattern (always ``1`` for exact paths).
    """
    patterns: dict[str, int] = {}
    exact: list[str] = []
    non_rigid_names: set[str] = set()
    for path in paths:
        if re.search(r"/World/envs/env_\d+/", path):
            non_rigid_names.add(path.rsplit("/", 1)[-1])

    for path in paths:
        body_name = path.rsplit("/", 1)[-1]
        if body_name in non_rigid_names and re.search(r"/World/envs/env_\d+/", path):
            wildcard = re.sub(r"/World/envs/env_\d+", "/World/envs/env_*", path)
            if wildcard != path:
                patterns[wildcard] = patterns.get(wildcard, 0) + 1
                continue
        exact.append(path)

    ordered_patterns = sorted(patterns.keys())
    pattern_counts = [patterns[pattern] for pattern in ordered_patterns]
    return [*ordered_patterns, *exact], pattern_counts


def path_to_env_wildcard(path: str) -> str:
    """Rewrite ``env_<id>`` segments to ``env_*`` for PhysX tensor patterns."""
    return re.sub(r"/World/envs/env_\d+", "/World/envs/env_*", path)


def path_to_env_regex(path: str) -> str:
    """Rewrite ``env_<id>`` segments to ``env_.*`` for Isaac Lab asset regex paths."""
    return re.sub(r"/World/envs/env_\d+", "/World/envs/env_.*", path)


def build_deformable_vertex_count_lookup(entries: list[DeformableStageEntry]) -> dict[str, int]:
    """Map deformable root and simulation-mesh paths to unpadded vertex counts."""
    path_to_count: dict[str, int] = {}
    for entry in entries:
        path_to_count[entry.root_path] = entry.vertex_count
        path_to_count[entry.sim_mesh_path] = entry.vertex_count
        path_to_count[entry.vis_mesh_path] = entry.vertex_count
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
