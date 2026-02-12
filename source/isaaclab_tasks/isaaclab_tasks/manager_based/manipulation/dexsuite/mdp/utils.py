# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import logging
import numpy as np
import torch
import trimesh
from trimesh.sample import sample_surface

from pxr import UsdGeom

import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim.utils import get_all_matching_child_prims

# ---- module-scope caches ----
_PRIM_SAMPLE_CACHE: dict[tuple[str, int], np.ndarray] = {}  # (prim_hash, num_points) -> (N,3) in root frame
_FINAL_SAMPLE_CACHE: dict[str, np.ndarray] = {}  # env_hash -> (num_points,3) in root frame


def clear_pointcloud_caches():
    _PRIM_SAMPLE_CACHE.clear()
    _FINAL_SAMPLE_CACHE.clear()


def sample_object_point_cloud(num_envs: int, num_points: int, prim_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Samples point clouds for each environment instance by collecting points
    from all matching USD prims under `prim_path`, then downsamples to
    exactly `num_points` per env using farthest-point sampling.

    Caching is in-memory within this module:
      - per-prim raw samples:   _PRIM_SAMPLE_CACHE[(prim_hash, num_points)]
      - final downsampled env:  _FINAL_SAMPLE_CACHE[env_hash]

    Returns:
        torch.Tensor: Shape (num_envs, num_points, 3) on `device`.
    """
    points = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)
    xform_cache = UsdGeom.XformCache()

    for i in range(num_envs):
        # Resolve prim path
        obj_path = prim_path.replace(".*", str(i))

        # Gather prims
        prims = get_all_matching_child_prims(
            obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
        )
        if not prims:
            raise KeyError(f"No valid prims under {obj_path}")

        object_prim = prim_utils.get_prim_at_path(obj_path)
        world_root = xform_cache.GetLocalToWorldTransform(object_prim)

        # hash each child prim by its rel transform + geometry
        prim_hashes = []
        for prim in prims:
            prim_type = prim.GetTypeName()
            hasher = hashlib.sha256()

            rel = world_root.GetInverse() * xform_cache.GetLocalToWorldTransform(prim)  # prim -> root
            mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
            hasher.update(mat_np.tobytes())

            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                hasher.update(verts.tobytes())
            else:
                if prim_type == "Cube":
                    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                    hasher.update(np.float32(size).tobytes())
                elif prim_type == "Sphere":
                    r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                    hasher.update(np.float32(r).tobytes())
                elif prim_type == "Cylinder":
                    c = UsdGeom.Cylinder(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Capsule":
                    c = UsdGeom.Capsule(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Cone":
                    c = UsdGeom.Cone(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())

            prim_hashes.append(hasher.hexdigest())

        # scale on root (default to 1 if missing)
        attr = object_prim.GetAttribute("xformOp:scale")
        scale_val = attr.Get() if attr else None
        if scale_val is None:
            base_scale = torch.ones(3, dtype=torch.float32, device=device)
        else:
            base_scale = torch.tensor(scale_val, dtype=torch.float32, device=device)

        # env-level cache key (includes num_points)
        env_key = "_".join(sorted(prim_hashes)) + f"_{num_points}"
        env_hash = hashlib.sha256(env_key.encode()).hexdigest()

        # load from env-level in-memory cache
        if env_hash in _FINAL_SAMPLE_CACHE:
            arr = _FINAL_SAMPLE_CACHE[env_hash]  # (num_points,3) in root frame
            points[i] = torch.from_numpy(arr).to(device) * base_scale.unsqueeze(0)
            continue

        # otherwise build per-prim samples (with per-prim cache)
        all_samples_np: list[np.ndarray] = []
        for prim, ph in zip(prims, prim_hashes):
            key = (ph, num_points)
            if key in _PRIM_SAMPLE_CACHE:
                samples = _PRIM_SAMPLE_CACHE[key]
            else:
                prim_type = prim.GetTypeName()
                if prim_type == "Mesh":
                    mesh = UsdGeom.Mesh(prim)
                    verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                    faces = _triangulate_faces(prim)
                    mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                else:
                    mesh_tm = create_primitive_mesh(prim)

                face_weights = mesh_tm.area_faces
                samples_np, _ = sample_surface(mesh_tm, num_points * 2, face_weight=face_weights)

                # FPS to num_points on chosen device
                tensor_pts = torch.from_numpy(samples_np.astype(np.float32)).to(device)
                prim_idxs = farthest_point_sampling(tensor_pts, num_points)
                local_pts = tensor_pts[prim_idxs]

                # prim -> root transform
                rel = xform_cache.GetLocalToWorldTransform(prim) * world_root.GetInverse()
                mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
                mat_t = torch.from_numpy(mat_np).to(device)

                ones = torch.ones((num_points, 1), device=device)
                pts_h = torch.cat([local_pts, ones], dim=1)
                root_h = pts_h @ mat_t
                samples = root_h[:, :3].detach().cpu().numpy()

                if prim_type == "Cone":
                    samples[:, 2] -= UsdGeom.Cone(prim).GetHeightAttr().Get() / 2

                _PRIM_SAMPLE_CACHE[key] = samples  # cache in root frame @ num_points

            all_samples_np.append(samples)

        # combine & env-level FPS (if needed)
        if len(all_samples_np) == 1:
            samples_final = torch.from_numpy(all_samples_np[0]).to(device)
        else:
            combined = torch.from_numpy(np.concatenate(all_samples_np, axis=0)).to(device)
            idxs = farthest_point_sampling(combined, num_points)
            samples_final = combined[idxs]

        # store env-level cache in root frame (CPU)
        _FINAL_SAMPLE_CACHE[env_hash] = samples_final.detach().cpu().numpy()

        # apply root scale and write out
        points[i] = samples_final * base_scale.unsqueeze(0)

    return points


def _triangulate_faces(prim) -> np.ndarray:
    """Convert a USD Mesh prim into triangulated face indices (N, 3)."""
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
    """Create a trimesh mesh from a USD primitive (Cube, Sphere, Cylinder, etc.)."""
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
        return trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Cone":  # Cone
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def farthest_point_sampling(
    points: torch.Tensor, n_samples: int, memory_threashold=2 * 1024**3
) -> torch.Tensor:  # 2 GiB
    """
    Farthest Point Sampling (FPS) for point sets.

    Selects `n_samples` points such that each new point is farthest from the
    already chosen ones. Uses a full pairwise distance matrix if memory allows,
    otherwise falls back to an iterative version.

    Args:
        points (torch.Tensor): Input points of shape (N, D).
        n_samples (int): Number of samples to select.
        memory_threashold (int): Max allowed bytes for distance matrix. Default 2 GiB.

    Returns:
        torch.Tensor: Indices of sampled points (n_samples,).
    """
    device = points.device
    N = points.shape[0]
    elem_size = points.element_size()
    bytes_needed = N * N * elem_size
    if bytes_needed <= memory_threashold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float("inf"), device=device)
        farthest = torch.randint(0, N, (1,), device=device)
        for j in range(n_samples):
            sampled_idx[j] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx
    logging.warning(f"FPS fallback to iterative (needed {bytes_needed} > {memory_threashold})")
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx
