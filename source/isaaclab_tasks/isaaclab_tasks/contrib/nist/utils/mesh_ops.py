# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from .rigid_object_hasher import RigidObjectHasher

if TYPE_CHECKING:
    import trimesh


def _gather_farthest_points(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Select a spatially spread-out subset of each point set.

    Starting from the first point of every set, repeatedly takes the candidate whose distance
    to the already-selected points is largest. The greedy choice gives near-uniform coverage of
    the input's spatial extent, which is what makes the returned cloud usable for the
    signed-distance collision queries downstream. Starting deterministically keeps the result
    reproducible without touching the global random state.

    Args:
        points: Candidate point sets [m], shape ``(B, N, D)``.
        num_samples: Number of points to select from each set. Clamped to ``N``.

    Returns:
        Selected points [m], shape ``(B, min(num_samples, N), D)``.
    """
    if points.ndim != 3:
        raise ValueError(f"Expected points of shape (B, N, D), got {tuple(points.shape)}.")
    num_sets, num_candidates, _ = points.shape
    num_samples = min(num_samples, num_candidates)

    indices = torch.empty(num_sets, num_samples, dtype=torch.long, device=points.device)
    # squared distances suffice: the square root is monotonic and only the arg-maximum is read
    min_sq_distance = torch.full((num_sets, num_candidates), torch.inf, dtype=points.dtype, device=points.device)
    set_ids = torch.arange(num_sets, device=points.device)
    selected = torch.zeros(num_sets, dtype=torch.long, device=points.device)
    for i in range(num_samples):
        indices[:, i] = selected
        sq_distance = (points - points[set_ids, selected].unsqueeze(1)).pow(2).sum(dim=-1)
        min_sq_distance = torch.minimum(min_sq_distance, sq_distance)
        selected = min_sq_distance.argmax(dim=1)
    return torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def _load_mesh_tensors(prim):
    tm = prim_to_trimesh(prim)
    verts = torch.from_numpy(tm.vertices.astype("float32"))
    faces = torch.from_numpy(tm.faces.astype("int64"))
    return verts, faces


@wp.kernel
def _sample_triangle_mesh_surface_kernel(
    verts: wp.array(dtype=wp.vec3),
    tris: wp.array2d(dtype=wp.int32),
    area_cdf: wp.array(dtype=wp.float32),
    tri_begin: wp.array(dtype=wp.int32),
    num_samples: int,
    seed: int,
    out: wp.array2d(dtype=wp.vec3),
):
    mesh_id, sample_id = wp.tid()
    begin = int(tri_begin[mesh_id])
    end = int(tri_begin[mesh_id + 1])
    if end <= begin:
        out[mesh_id, sample_id] = wp.vec3(0.0, 0.0, 0.0)
        return

    rng = wp.rand_init(seed, mesh_id * num_samples + sample_id)
    target = wp.randf(rng)

    # first triangle whose per-mesh cumulative area reaches ``target``, so a triangle is
    # picked with probability proportional to its share of the mesh surface
    lo = begin
    hi = end - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if target <= area_cdf[mid]:
            hi = mid
        else:
            lo = mid + 1

    bary = wp.sample_triangle(rng)
    out[mesh_id, sample_id] = (
        bary[0] * verts[tris[lo, 0]] + bary[1] * verts[tris[lo, 1]] + (1.0 - bary[0] - bary[1]) * verts[tris[lo, 2]]
    )


def sample_triangle_mesh_surface(
    meshes: Sequence[tuple[torch.Tensor, torch.Tensor]],
    num_samples: int,
    device: str,
    seed: int = 42,
) -> torch.Tensor:
    """Sample points uniformly over the surface area of a batch of triangle meshes.

    Every sample picks a triangle with probability proportional to its area, then draws a
    uniform barycentric point inside it. The meshes may differ in vertex and triangle
    count: they are packed into flat Warp arrays and sampled by a single kernel launch.
    Sampling is reproducible for a given :paramref:`seed` and leaves the global random
    state untouched.

    Args:
        meshes: One ``(vertices, triangles)`` pair per mesh, holding vertex positions [m] of
            shape ``(V, 3)`` and triangle vertex indices of shape ``(F, 3)``. Meshes without
            triangles yield samples at the origin.
        num_samples: Number of surface points to draw from every mesh.
        device: Device on which to sample.
        seed: Seed for the sampling.

    Returns:
        Sampled surface points [m] in each mesh's own frame, shape ``(len(meshes), num_samples, 3)``.
    """
    num_meshes = len(meshes)
    verts_list: list[torch.Tensor] = []
    tris_list: list[torch.Tensor] = []
    vertex_offset = 0
    for verts, tris in meshes:
        verts_list.append(verts)
        tris_list.append(tris + vertex_offset)
        vertex_offset += verts.shape[0]

    verts_packed = torch.cat(verts_list).to(device=device, dtype=torch.float32).contiguous()
    tris_packed = torch.cat(tris_list).to(device=device, dtype=torch.int64)
    tri_begin = torch.zeros(num_meshes + 1, dtype=torch.int64, device=device)
    tri_begin[1:] = torch.tensor([tris.shape[0] for _, tris in meshes], device=device).cumsum(0)

    corners = verts_packed[tris_packed]
    areas = 0.5 * torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]).norm(dim=-1)
    # normalize the cumulative areas per mesh so the kernel can binary-search a plain
    # uniform draw within each mesh's triangle range without a segment rescale
    inclusive = torch.cat([areas.new_zeros(1, dtype=torch.float64), areas.double().cumsum(0)])
    base = inclusive[tri_begin[:-1]]
    total = (inclusive[tri_begin[1:]] - base).clamp(min=torch.finfo(torch.float64).tiny)
    mesh_of_tri = torch.repeat_interleave(torch.arange(num_meshes, device=device), tri_begin[1:] - tri_begin[:-1])
    area_cdf = ((inclusive[1:] - base[mesh_of_tri]) / total[mesh_of_tri]).float().contiguous()

    # the Warp views below alias these tensors, so keep them bound for the whole launch
    tris_i32 = tris_packed.to(torch.int32).contiguous()
    tri_begin_i32 = tri_begin.to(torch.int32).contiguous()
    points = torch.empty((num_meshes, num_samples, 3), dtype=torch.float32, device=device)
    wp.launch(
        _sample_triangle_mesh_surface_kernel,
        dim=(num_meshes, num_samples),
        inputs=[
            wp.from_torch(verts_packed, dtype=wp.vec3),
            wp.from_torch(tris_i32, dtype=wp.int32),
            wp.from_torch(area_cdf, dtype=wp.float32),
            wp.from_torch(tri_begin_i32, dtype=wp.int32),
            num_samples,
            seed,
        ],
        outputs=[wp.from_torch(points, dtype=wp.vec3)],
        device=str(device),
    )
    return points


def sample_object_point_cloud(
    num_envs: int,
    num_points: int,
    prim_path_pattern: str,
    device: str = "cuda",
    rigid_object_hasher: RigidObjectHasher | None = None,
    seed: int = 42,
    *,
    include_boundary_points: bool = False,
) -> torch.Tensor | None:
    """Sample a point cloud on the collision geometry of an asset.

    Robust to heterogeneous collider counts across envs. Uses
    ``RigidObjectHasher`` to reuse identical ordered geometry batches and sampling modes.

    Args:
        num_envs: Number of environments to sample.
        num_points: Number of points per asset root.
        prim_path_pattern: Asset root path expression.
        device: Device on which to sample.
        rigid_object_hasher: Existing geometry index, or ``None`` to construct one.
        seed: Seed for surface sampling.
        include_boundary_points: Include a regular boundary grid for single-cube bodies when the
            budget permits eight points. Single-mesh bodies reserve at most half the budget for
            unique surface vertices, spread out when they do not all fit. Collision checks can opt
            in to retain these boundaries; the default preserves observation surface sampling.

    Returns:
        Root-frame surface points [m], shape ``(num_envs, num_points, 3)``, or ``None`` without geometry.
    """
    hasher = (
        rigid_object_hasher
        if rigid_object_hasher is not None
        else RigidObjectHasher(num_envs, prim_path_pattern, device=device)
    )

    if hasher.num_root == 0 or not hasher.collider_prims:
        return None

    replicated_env = torch.all(hasher.root_prim_hashes == hasher.root_prim_hashes[0])
    hashes = hasher.root_prim_hashes[:1] if replicated_env else hasher.root_prim_hashes
    cache_device = str(torch.empty(0, device=device).device)
    cache_key = (tuple(hashes.tolist()), seed, num_points, cache_device, include_boundary_points)
    point_cloud_store = hasher.point_cloud_cache
    root_scales = hasher.root_prim_scales.to(device)[:, None]
    if cache_key in point_cloud_store:
        merged = point_cloud_store[cache_key]
        return merged.expand(num_envs if replicated_env else hasher.num_root, -1, -1) * root_scales

    if replicated_env:
        mask_env0 = hasher.collider_prim_env_ids == 0
        prims = [p for p, m in zip(hasher.collider_prims, mask_env0) if m]
        env_ids = hasher.collider_prim_env_ids[mask_env0]
        rel_pos = hasher.collider_rel_pos[mask_env0]
        rel_mat = hasher.collider_rel_mat[mask_env0]
    else:
        prims = hasher.collider_prims
        env_ids = hasher.collider_prim_env_ids
        rel_pos = hasher.collider_rel_pos
        rel_mat = hasher.collider_rel_mat
    mesh_tensors = [_load_mesh_tensors(p) for p in prims]

    # oversample each collider, then thin to ``num_points`` so the kept points spread over
    # the whole collider rather than clustering on its largest faces
    surface = sample_triangle_mesh_surface(mesh_tensors, num_points * 2, device=device, seed=seed)
    local = _gather_farthest_points(surface, num_points)
    boundary_colliders = []
    if include_boundary_points and num_points >= 2:
        counts = torch.bincount(env_ids, minlength=hasher.num_root).tolist()
        rows = env_ids.tolist()
        singles = [i for i, row in enumerate(rows) if counts[row] == 1]
        cubes = [i for i in singles if prims[i].GetTypeName() == "Cube" and num_points >= 8]
        if cubes:
            from pxr import UsdGeom

            resolution = 2
            while (resolution + 1) ** 3 - (resolution - 1) ** 3 <= num_points:
                resolution += 1
            axis = torch.linspace(-1.0, 1.0, resolution, device=device)
            grid = torch.cartesian_prod(axis, axis, axis)
            grid = grid[(grid.abs() == 1.0).any(dim=1)]
            for i in cubes:
                local[i, : len(grid)] = grid * (0.5 * UsdGeom.Cube(prims[i]).GetSizeAttr().Get())
            boundary_colliders.extend(cubes)
        for i in singles:
            if prims[i].GetTypeName() == "Mesh":
                vertices, faces = mesh_tensors[i]
                # Ignore unused vertices: they are not part of the collision surface.
                vertices = torch.unique(vertices[torch.unique(faces)], dim=0).to(device)
                if len(vertices) > num_points // 2:
                    vertices = _gather_farthest_points(vertices.unsqueeze(0), num_points // 2)[0]
                local[i, : len(vertices)] = vertices
                boundary_colliders.append(i)
    root = torch.einsum("nij,npj->npi", rel_mat.to(device), local) + rel_pos.to(device).unsqueeze(1)

    if len(boundary_colliders) == len(prims) == len(hashes):
        merged = torch.empty_like(root)
        merged[env_ids.to(device)] = root
    elif replicated_env:
        merged = _gather_farthest_points(root.reshape(1, -1, 3), num_points)
    else:
        env_ids = hasher.collider_prim_env_ids.to(device)
        counts = torch.bincount(env_ids, minlength=hasher.num_root)
        if (counts == 0).any():
            raise ValueError("Every heterogeneous asset root must contain collision geometry.")
        max_c = int(counts.max().item())
        buf = torch.empty((hasher.num_root, max_c * num_points, 3), device=device, dtype=root.dtype)
        placed = torch.zeros_like(counts)
        for i in range(len(hasher.collider_prims)):
            r = int(env_ids[i].item())
            start = placed[r].item() * num_points
            if start == 0:
                # Padding must duplicate a surface point, not introduce the asset origin.
                buf[r] = root[i, 0]
            buf[r, start : start + num_points] = root[i]
            placed[r] += 1
        merged = _gather_farthest_points(buf, num_points)
        if boundary_colliders:
            # Compound-body thinning must not discard a neighboring single body's boundary samples.
            merged[env_ids[boundary_colliders]] = root[boundary_colliders]

    point_cloud_store[cache_key] = merged
    return merged.expand(num_envs if replicated_env else hasher.num_root, -1, -1) * root_scales


def _triangulate_faces(prim) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim) -> trimesh.Trimesh:
    import trimesh
    from trimesh.transformations import rotation_matrix

    from pxr import UsdGeom

    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        tri_mesh = trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
        if c.GetAxisAttr().Get() == "X":
            R = rotation_matrix(np.radians(-90), [0, 1, 0])
            tri_mesh.apply_transform(R)
        elif c.GetAxisAttr().Get() == "Y":
            R = rotation_matrix(np.radians(90), [1, 0, 0])
            tri_mesh.apply_transform(R)
        return tri_mesh
    elif prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        radius = c.GetRadiusAttr().Get()
        height = c.GetHeightAttr().Get()
        mesh = trimesh.creation.cone(radius=radius, height=height)
        mesh.apply_translation((0.0, 0.0, -height / 2.0))
        return mesh
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def prim_to_trimesh(prim, relative_to_world=False) -> trimesh.Trimesh:
    import trimesh

    from pxr import Usd, UsdGeom

    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        mesh_tm = create_primitive_mesh(prim)

    if relative_to_world:
        tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        mesh_tm.apply_transform(tf)

    return mesh_tm


def prim_to_warp_mesh(
    prim,
    device,
    relative_to_world=False,
    merge_prims: Sequence | None = None,
    linear_transform: np.ndarray | None = None,
) -> wp.Mesh:
    """Build a winding-enabled mesh, optionally merging parts and baking a local linear transform.

    Args:
        prim: Source USD geometry prim, also defining the local frame for merged parts.
        device: Device on which to create the mesh.
        relative_to_world: Whether to apply the prim's world transform after local geometry transforms.
        merge_prims: Parts to merge in ``prim`` coordinates, or ``None`` to use only ``prim``.
        linear_transform: Static matrix of shape ``(3, 3)`` applied after merging. Reflections
            preserve outward winding by reversing triangle order.

    Returns:
        Mesh whose nearest-point queries use the metric of the transformed vertices.
    """
    from pxr import Usd, UsdGeom

    source_prims = (prim,) if merge_prims is None else merge_prims
    if not source_prims:
        raise ValueError("At least one source prim is required to construct a collision mesh.")
    xform_cache = UsdGeom.XformCache() if merge_prims is not None else None
    target_inv = xform_cache.GetLocalToWorldTransform(prim).GetInverse() if xform_cache is not None else None
    point_sets, triangle_sets = [], []
    vertex_offset = 0
    for source_prim in source_prims:
        if source_prim.GetTypeName() == "Mesh":
            points = np.asarray(UsdGeom.Mesh(source_prim).GetPointsAttr().Get(), dtype=np.float32)
            indices = _triangulate_faces(source_prim).astype(np.int32)
        else:
            mesh = create_primitive_mesh(source_prim)
            points = mesh.vertices.astype(np.float32)
            indices = mesh.faces.astype(np.int32)
        if xform_cache is not None and source_prim != prim:
            transform = np.array(xform_cache.GetLocalToWorldTransform(source_prim) * target_inv).T
            points = (points @ transform[:3, :3].T) + transform[:3, 3]
            if np.linalg.det(transform[:3, :3]) < 0.0:
                indices = indices[:, [0, 2, 1]]
        point_sets.append(points)
        triangle_sets.append(indices + vertex_offset)
        vertex_offset += len(points)
    points = np.concatenate(point_sets)
    indices = np.concatenate(triangle_sets)
    if linear_transform is not None:
        points = points @ linear_transform.T
        if np.linalg.det(linear_transform) < 0.0:
            indices = indices[:, [0, 2, 1]]

    if relative_to_world:
        tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        points = (points @ tf[:3, :3].T) + tf[:3, 3]

    return wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices.flatten(), dtype=wp.int32, device=device),
        support_winding_number=True,
    )
