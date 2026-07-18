# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from trimesh.sample import sample_surface

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

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
    # Obtain stage handle
    stage = sim_utils.get_current_stage()

    sample_targets: list[tuple[str, tuple[int, ...]]] = []
    clone_plan = sim_utils.SimulationContext.instance().get_clone_plan()
    for _, _, source_path, env_ids in iter_clone_plan_matches(clone_plan, prim_path):
        sample_targets.append((source_path, env_ids))

    for obj_path, env_ids in sample_targets:
        # Gather prims
        prims = sim_utils.get_all_matching_child_prims(
            obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
        )
        if not prims:
            raise KeyError(f"No valid prims under {obj_path}")

        object_prim = stage.GetPrimAtPath(obj_path)
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
            scaled_samples = torch.from_numpy(arr).to(device) * base_scale.unsqueeze(0)
            for env_id in env_ids:
                points[env_id] = scaled_samples
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
        scaled_samples = samples_final * base_scale.unsqueeze(0)
        for env_id in env_ids:
            points[env_id] = scaled_samples

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


def create_primitive_mesh(prim):
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
        dist = torch.linalg.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def collect_collision_meshes(root_prim, owner_frame_fn: Callable) -> dict[int, trimesh.Trimesh]:
    """Collect collision meshes under ``root_prim``, grouped in caller-selected frames."""
    mesh_types = PRIMITIVE_MESH_TYPES + ["Mesh"]
    mesh_prims = sim_utils.get_all_matching_child_prims(
        root_prim.GetPath(),
        lambda prim: prim.GetTypeName() in mesh_types and prim.HasAPI(UsdPhysics.CollisionAPI),
    )

    meshes_by_owner: dict[int, list[trimesh.Trimesh]] = {}
    for prim in mesh_prims:
        owner_frame = owner_frame_fn(prim)
        if owner_frame is None:
            continue
        owner, frame_prim = owner_frame
        mesh = (
            create_trimesh_from_geom_mesh(prim)
            if prim.GetTypeName() == "Mesh"
            else create_trimesh_from_geom_shape(prim)
        )
        mesh.apply_scale(sim_utils.resolve_prim_scale(prim))
        position, quat_xyzw = sim_utils.resolve_prim_pose(prim, frame_prim)
        transform = trimesh.transformations.quaternion_matrix([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        transform[:3, 3] = position
        mesh.apply_transform(transform)
        meshes_by_owner.setdefault(owner, []).append(mesh)

    return {
        owner: trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        for owner, meshes in meshes_by_owner.items()
    }


def collect_body_collision_meshes(robot, body_names: str | list[str]) -> tuple[dict[int, trimesh.Trimesh], list[str]]:
    """Extract the selected bodies' collision meshes from the env_0 clone, baked body-local.

    Returns ``(body_meshes, body_names)`` where ``body_meshes`` maps a body index to one
    merged :class:`trimesh.Trimesh` in that body's frame.
    """
    body_ids, names = robot.find_bodies(body_names)
    body_id_of = dict(zip(names, body_ids))
    robot_prim = sim_utils.find_matching_prims(robot.cfg.prim_path)[0]

    def body_frame(prim):
        # links nest in converted assets: the nearest selected-body ancestor owns the collider
        node = prim.GetParent()
        while node.IsValid() and node.GetName() not in body_id_of:
            node = node.GetParent()
        if not node.IsValid():
            return None
        return body_id_of[node.GetName()], node

    body_meshes = collect_collision_meshes(robot_prim, body_frame)
    if not body_meshes:
        raise RuntimeError(f"no collision meshes found under '{robot.cfg.prim_path}' for bodies {names}.")
    return body_meshes, names


def get_reset_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_assets: Sequence[str],
    is_relative: bool = False,
) -> torch.Tensor:
    """Read and concatenate the reset-state slices of the given scene assets.

    Per articulation: root pose [m, (x, y, z, w)] (7), root center-of-mass velocity
    [m/s, rad/s] (6), joint positions and joint velocities; per rigid object: root pose and
    root center-of-mass velocity. With :paramref:`is_relative`, root positions are expressed
    relative to the environment origins so states transplant across environments.
    """

    def root_state(asset) -> list[torch.Tensor]:
        pose = asset.data.root_link_pose_w.torch[env_ids]
        if is_relative:
            pose = pose.clone()
            pose[:, :3] -= env.scene.env_origins[env_ids]
        return [pose, asset.data.root_com_vel_w.torch[env_ids]]

    states: list[torch.Tensor] = []
    for name, articulation in env.scene.articulations.items():
        if name in reset_assets:
            states += root_state(articulation)
            states.append(articulation.data.joint_pos.torch[env_ids])
            states.append(articulation.data.joint_vel.torch[env_ids])
    for name, rigid_object in env.scene.rigid_objects.items():
        if name in reset_assets:
            states += root_state(rigid_object)
    return torch.cat(states, dim=-1)


def set_reset_state(
    env: ManagerBasedEnv,
    states: torch.Tensor,
    env_ids: torch.Tensor,
    reset_assets: Sequence[str],
    is_relative: bool = False,
):
    """Split :paramref:`states` by scene asset and write the reset-state slices.

    Inverse of :func:`get_reset_state`; the layout and :paramref:`is_relative` convention
    must match the call that produced :paramref:`states`.
    """
    offset = 0

    def write_root(asset):
        nonlocal offset
        pose = states[:, offset : offset + 7].clone()
        if is_relative:
            pose[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_com_velocity_to_sim_index(
            root_velocity=states[:, offset + 7 : offset + 13].contiguous(), env_ids=env_ids
        )
        offset += 13

    for name, articulation in env.scene.articulations.items():
        if name in reset_assets:
            write_root(articulation)
            num_joints = articulation.num_joints
            articulation.write_joint_state_to_sim_index(
                position=states[:, offset : offset + num_joints].contiguous(),
                velocity=states[:, offset + num_joints : offset + 2 * num_joints].contiguous(),
                env_ids=env_ids,
            )
            offset += 2 * num_joints
    for name, rigid_object in env.scene.rigid_objects.items():
        if name in reset_assets:
            write_root(rigid_object)
