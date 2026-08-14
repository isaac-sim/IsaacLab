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

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.contrib.nist.utils.rigid_object_hasher import RigidObjectHasher

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
) -> torch.Tensor | None:
    """Sample a point cloud on the collision geometry of an asset.

    Robust to heterogeneous collider counts across envs. Uses
    ``RigidObjectHasher`` to deduplicate identical colliders.
    """
    hasher = rigid_object_hasher if rigid_object_hasher is not None else RigidObjectHasher(num_envs, prim_path_pattern)

    if hasher.num_root == 0:
        return None

    replicated_env = torch.all(hasher.root_prim_hashes == hasher.root_prim_hashes[0])
    if replicated_env:
        mask_env0 = hasher.collider_prim_env_ids == 0
        mesh_tensors = [_load_mesh_tensors(p) for p, m in zip(hasher.collider_prims, mask_env0) if m]
        rel_pos = hasher.collider_rel_pos[mask_env0]
        rel_mat = hasher.collider_rel_mat[mask_env0]
    else:
        mesh_tensors = [_load_mesh_tensors(p) for p in hasher.collider_prims]
        rel_pos = hasher.collider_rel_pos
        rel_mat = hasher.collider_rel_mat

    # oversample each collider, then thin to ``num_points`` so the kept points spread over
    # the whole collider rather than clustering on its largest faces
    surface = sample_triangle_mesh_surface(mesh_tensors, num_points * 2, device=device, seed=seed)
    local = _gather_farthest_points(surface, num_points)
    root = torch.einsum("nij,npj->npi", rel_mat.to(device), local) + rel_pos.to(device).unsqueeze(1)

    if replicated_env:
        merged = _gather_farthest_points(root.reshape(1, -1, 3), num_points)
        result = merged.expand(num_envs, -1, -1) * hasher.root_prim_scales.to(device).unsqueeze(1)
    else:
        env_ids = hasher.collider_prim_env_ids.to(device)
        counts = torch.bincount(env_ids, minlength=hasher.num_root)
        max_c = int(counts.max().item())
        buf = torch.zeros((hasher.num_root, max_c * num_points, 3), device=device, dtype=root.dtype)
        placed = torch.zeros_like(counts)
        for i in range(len(hasher.collider_prims)):
            r = int(env_ids[i].item())
            start = placed[r].item() * num_points
            buf[r, start : start + num_points] = root[i]
            placed[r] += 1
        merged = _gather_farthest_points(buf, num_points)
        result = merged * hasher.root_prim_scales.to(device).unsqueeze(1)

    return result


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

    import omni
    from pxr import UsdGeom

    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        mesh_tm = create_primitive_mesh(prim)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T
        mesh_tm.apply_transform(tf)

    return mesh_tm


def prim_to_warp_mesh(prim, device, relative_to_world=False) -> wp.Mesh:
    from pxr import UsdGeom

    if prim.GetTypeName() == "Mesh":
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    else:
        mesh = create_primitive_mesh(prim)
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)

    if relative_to_world:
        import omni

        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T
        points = (points @ tf[:3, :3].T) + tf[:3, 3]

    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    return wp_mesh
