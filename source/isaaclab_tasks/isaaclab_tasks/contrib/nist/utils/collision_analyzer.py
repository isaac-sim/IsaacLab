# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.cloner import num_spawn_variants
from isaaclab.sim.utils import resolve_matching_prims_from_source

from . import mesh_ops as _mesh_ops
from .rigid_object_hasher import RigidObjectHasher
from .sdf import (
    ColliderTransform,
    _pack_collider_bounds,
    get_signed_distance_mega,
    pack_collider_transforms,
    update_articulated_collider_transforms,
    update_root_poses,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .collision_analyzer_cfg import CollisionAnalyzerCfg


class CollisionAnalyzer:
    cfg: CollisionAnalyzerCfg

    def __init__(self, cfg: CollisionAnalyzerCfg, env: ManagerBasedRLEnv):
        from pxr import Gf, UsdGeom, UsdPhysics

        self.cfg = cfg
        self.asset: RigidObject = env.scene[cfg.asset_cfg.name]
        self.obstacles: list[RigidObject] = [env.scene[cfg.name] for cfg in cfg.obstacle_cfgs]
        device = env.device
        min_dist_by_obstacle = cfg.min_dist_by_obstacle
        obstacle_names = [obstacle_cfg.name for obstacle_cfg in cfg.obstacle_cfgs]
        unknown_obstacles = min_dist_by_obstacle.keys() - set(obstacle_names)
        if unknown_obstacles:
            raise ValueError(f"Unknown obstacle minimum-distance overrides: {sorted(unknown_obstacles)}.")
        min_dists = [min_dist_by_obstacle.get(name, cfg.min_dist) for name in obstacle_names]
        if not all(np.isfinite(value) and value <= cfg.max_dist for value in min_dists):
            raise ValueError("Obstacle minimum distances must be finite and no greater than max_dist.")
        self._check_dist = any(value != 0.0 for value in min_dists)
        self._min_dist_offsets = torch.tensor([cfg.min_dist - value for value in min_dists], device=device)
        self.wp_min_dist_offsets = wp.from_torch(self._min_dist_offsets, dtype=wp.float32)
        if cfg.asset_cfg.body_names is None:
            body_ids = np.arange(len(self.asset.body_names))[cfg.asset_cfg.body_ids].tolist()
            body_names = [self.asset.body_names[body_id] for body_id in body_ids]
        else:
            body_ids, body_names = self.asset.find_bodies(
                cfg.asset_cfg.body_names, preserve_order=cfg.asset_cfg.preserve_order
            )

        self.body_ids = []
        point_clouds = []
        num_asset_variants = num_spawn_variants(self.asset.cfg.spawn)
        if num_asset_variants > 1:
            asset_hasher = RigidObjectHasher(
                env.num_envs,
                self.asset.cfg.prim_path,
                device=device,
                rigid_body_root=True,
                compact_sources=True,
                source_paths=self.asset.cfg.spawn.spawn_paths,
                geometry_policy=getattr(cfg.asset_cfg, "geometry_policy", "authored_collision"),
            )
            if asset_hasher.num_root != num_asset_variants:
                raise ValueError(
                    f"Expected {num_asset_variants} collision sources below '{self.asset.cfg.prim_path}', found"
                    f" {asset_hasher.num_root}."
                )
            local_pts = _mesh_ops.sample_object_point_cloud(
                num_envs=num_asset_variants,
                num_points=cfg.num_points,
                device=device,
                prim_path_pattern=self.asset.cfg.prim_path,
                rigid_object_hasher=asset_hasher,
                include_boundary_points=True,
            )
            if local_pts is None:
                raise ValueError(f"No collision geometry found below '{self.asset.cfg.prim_path}'.")
            point_clouds.append(local_pts.unsqueeze(1))
            self.body_ids.append(0)
        else:
            for body_id, body_name in zip(body_ids, body_names):
                _, prim_path = resolve_matching_prims_from_source(
                    self.asset.cfg.prim_path,
                    predicate=lambda p, name=body_name: p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI),
                    expected_num_matches=1,
                )[0]
                asset_hasher = RigidObjectHasher(
                    env.num_envs,
                    prim_path,
                    device=device,
                    compact_sources=True,
                    geometry_policy=getattr(cfg.asset_cfg, "geometry_policy", "authored_collision"),
                    body_names=(body_name,),
                )
                if asset_hasher.num_root != 1:
                    raise ValueError(f"Collision query asset '{prim_path}' must have one clone source.")
                local_pts = _mesh_ops.sample_object_point_cloud(
                    num_envs=1,
                    num_points=cfg.num_points,
                    prim_path_pattern=prim_path,
                    device=device,
                    rigid_object_hasher=asset_hasher,
                    include_boundary_points=True,
                )
                if local_pts is not None:
                    point_clouds.append(local_pts.unsqueeze(1))
                    self.body_ids.append(body_id)
        if not point_clouds:
            raise ValueError(f"No collision geometry found below '{self.asset.cfg.prim_path}'.")
        self.local_pts = torch.cat(point_clouds, dim=1).contiguous()
        # Neighboring CUDA lanes traverse nearby mesh regions; only reorder, never thin the safety samples.
        lower = self.local_pts.amin(dim=-2, keepdim=True)
        span = self.local_pts.amax(dim=-2, keepdim=True) - lower
        coordinates = ((self.local_pts - lower) / torch.where(span > 0, span, 1.0) * 1023).long()
        # Interleave ten bits of each coordinate into a Morton key.
        coordinates = (coordinates | (coordinates << 16)) & 0x030000FF
        coordinates = (coordinates | (coordinates << 8)) & 0x0300F00F
        coordinates = (coordinates | (coordinates << 4)) & 0x030C30C3
        coordinates = (coordinates | (coordinates << 2)) & 0x09249249
        keys = coordinates[..., 0] | (coordinates[..., 1] << 1) | (coordinates[..., 2] << 2)
        order = keys.argsort(dim=-1, stable=True)
        self.local_pts = self.local_pts.gather(-2, order[..., None].expand(*order.shape, 3)).contiguous()
        self._body_ids_i32 = torch.tensor(self.body_ids, dtype=torch.int32, device=device)
        self._asset_uses_variants = num_asset_variants > 1
        self._asset_rows = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
        self.wp_asset_rows = wp.from_torch(self._asset_rows, dtype=wp.int32)

        num_obstacles = len(self.obstacles)
        num_envs = env.num_envs

        obstacle_row_counts: list[int] = []
        obstacle_prim_counts: list[torch.Tensor] = []
        obstacle_root_scales: list[torch.Tensor] = []

        # First pass: mesh dedup + collect flat CPU data per obstacle.
        # We store flat arrays and env_ids from the hasher, then scatter
        # into the strided layout after max_prims is known.
        obs_flat_pos: list[torch.Tensor] = []
        obs_flat_mat: list[torch.Tensor] = []
        obs_flat_mat_inv: list[torch.Tensor] = []
        obs_flat_env_ids: list[torch.Tensor] = []
        obs_flat_handles: list[list[int]] = []
        meshes: dict[int, wp.Mesh] = {}
        self._meshes: dict[tuple, wp.Mesh] = {}

        # Articulated obstacles (multiple body links): the hasher's root-relative
        # transforms freeze the spawn articulation, so also capture each collider's
        # owning body link and its static BODY-relative transform. The runtime pass
        # recomposes world transforms from live body poses instead of the root pose.
        art_flat: dict[int, dict[str, torch.Tensor]] = {}
        variant_obstacles: set[int] = set()

        for i, obstacle in enumerate(self.obstacles):
            start = time.perf_counter()
            num_variants = num_spawn_variants(obstacle.cfg.spawn)
            articulated = len(obstacle.body_names) > 1
            obstacle_cfg = cfg.obstacle_cfgs[i]
            selected_body_names = None
            if obstacle_cfg.body_names is not None:
                _, selected_body_names = obstacle.find_bodies(
                    obstacle_cfg.body_names, preserve_order=obstacle_cfg.preserve_order
                )
            elif obstacle_cfg.body_ids != slice(None):
                selected_body_names = np.asarray(obstacle.body_names)[obstacle_cfg.body_ids].tolist()
            obs_h = RigidObjectHasher(
                num_envs,
                prim_path_pattern=obstacle.cfg.prim_path,
                device="cpu",
                rigid_body_root=num_variants > 1,
                compact_sources=not articulated,
                source_paths=obstacle.cfg.spawn.spawn_paths if num_variants > 1 else None,
                geometry_policy=getattr(obstacle_cfg, "geometry_policy", "authored_collision"),
                body_names=selected_body_names,
            )
            if num_variants > 1:
                variant_obstacles.add(i)
            if not articulated and obs_h.num_root != num_variants:
                raise ValueError(
                    f"Expected {num_variants} collision sources below '{obstacle.cfg.prim_path}',"
                    f" found {obs_h.num_root}."
                )

            rel_pos_cpu = obs_h.collider_rel_pos
            rel_mat_cpu = obs_h.collider_rel_mat
            env_ids_cpu = obs_h.collider_prim_env_ids
            hashes_cpu = obs_h.collider_prim_hashes
            group_ids_cpu = obs_h.collider_prim_group_ids

            # Material-split visual surfaces define one solid per body, not independent interiors.
            groups: dict[tuple[int, int], list[int]] = {}
            for j, (row, group) in enumerate(zip(env_ids_cpu.tolist(), group_ids_cpu.tolist())):
                key = (row, group)
                groups.setdefault(key, []).append(j)
            anchors = torch.tensor([indices[0] for indices in groups.values()], dtype=torch.long)
            collider_prims = [obs_h.collider_prims[j] for j in anchors.tolist()]
            root_scales = obs_h.root_prim_scales
            collider_scales = root_scales[env_ids_cpu[anchors]]
            collider_pos = rel_pos_cpu[anchors] * collider_scales
            baked_mat = collider_scales[:, :, None] * rel_mat_cpu[anchors]
            identity = torch.eye(3).expand(len(anchors), -1, -1).clone()

            if articulated and collider_prims:
                xform_cache = UsdGeom.XformCache()
                per_prim: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
                body_ids_flat: list[int] = []
                rel_pos_body: list[torch.Tensor] = []
                rel_mat_body: list[torch.Tensor] = []
                for prim in collider_prims:
                    key = str(prim.GetPath())
                    cached = per_prim.get(key)
                    if cached is None:
                        link = prim
                        while link.IsValid() and not link.HasAPI(UsdPhysics.RigidBodyAPI):
                            link = link.GetParent()
                        if not link.IsValid() or link.GetName() not in obstacle.body_names:
                            raise RuntimeError(
                                f"Collider '{key}' of articulated obstacle '{obstacle.cfg.prim_path}' has no"
                                " rigid-body ancestor matching one of its body links."
                            )
                        # Live body poses contain only translation/rotation. Preserve all static
                        # scale and shear, including inheritance above the owning body, in its mesh.
                        link_xf = Gf.Transform(xform_cache.GetLocalToWorldTransform(link))
                        rigid_xf = Gf.Matrix4d(link_xf.GetRotation(), link_xf.GetTranslation())
                        rel4 = np.array(xform_cache.GetLocalToWorldTransform(prim) * rigid_xf.GetInverse())
                        cached = (
                            obstacle.body_names.index(link.GetName()),
                            torch.tensor(rel4[3, :3], dtype=torch.float32),
                            torch.tensor(rel4[:3, :3].T, dtype=torch.float32),
                        )
                        per_prim[key] = cached
                    body_ids_flat.append(cached[0])
                    rel_pos_body.append(cached[1])
                    rel_mat_body.append(cached[2])
                collider_pos = torch.stack(rel_pos_body)
                baked_mat = torch.stack(rel_mat_body)
                art_flat[i] = {
                    "body_ids": torch.tensor(body_ids_flat, dtype=torch.int32),
                    "rel_pos": collider_pos,
                    "rel_mat": identity,
                }

            handles_flat: list[int] = []
            for indices, prim, matrix in zip(groups.values(), collider_prims, baked_mat.numpy()):
                mesh_key = (tuple(hashes_cpu[indices].tolist()), matrix.tobytes(), str(wp.get_device(device)))
                if mesh_key not in self._meshes:
                    merge_prims = [obs_h.collider_prims[j] for j in indices] if len(indices) > 1 else None
                    self._meshes[mesh_key] = _mesh_ops.prim_to_warp_mesh(
                        prim, device=device, merge_prims=merge_prims, linear_transform=matrix
                    )
                mesh = self._meshes[mesh_key]
                handles_flat.append(mesh.id)
                meshes[mesh.id] = mesh

            obs_flat_pos.append(collider_pos)
            obs_flat_mat.append(identity)
            obs_flat_mat_inv.append(identity)
            obs_flat_env_ids.append(env_ids_cpu[anchors])
            obs_flat_handles.append(handles_flat)
            obstacle_row_counts.append(obs_h.num_root)
            obstacle_prim_counts.append(torch.bincount(env_ids_cpu[anchors], minlength=obs_h.num_root).to(torch.int32))
            obstacle_root_scales.append(torch.ones_like(root_scales, device=device))

            pc_time = time.perf_counter() - start
            print(f"Sampled {len(collider_prims)} wp meshes at '{obstacle.cfg.prim_path}' in {pc_time:.3f}s")

        max_prims = max(int(counts.max().item()) for counts in obstacle_prim_counts)
        self.max_prims = max_prims
        self.num_obstacles = num_obstacles

        row_offsets = torch.tensor([0] + obstacle_row_counts, dtype=torch.int64).cumsum(0)[:-1]
        total_rows = sum(obstacle_row_counts)
        total_slots = total_rows * max_prims
        pos_buf = torch.zeros(total_slots, 3)
        mat_buf = torch.zeros(total_slots, 3, 3)
        mat_inv_buf = torch.zeros(total_slots, 3, 3)
        handle_buf = torch.zeros(total_slots, dtype=torch.int64)

        for obs_idx in range(num_obstacles):
            collider_rows = obs_flat_env_ids[obs_idx]
            num_colliders = len(collider_rows)
            slot_offsets = torch.zeros(num_colliders, dtype=torch.long)
            placed_per_row = torch.zeros(obstacle_row_counts[obs_idx], dtype=torch.long)
            for c in range(num_colliders):
                row = collider_rows[c].item()
                slot_offsets[c] = placed_per_row[row]
                placed_per_row[row] += 1
            dst = (row_offsets[obs_idx] + collider_rows) * max_prims + slot_offsets
            pos_buf[dst] = obs_flat_pos[obs_idx]
            mat_buf[dst] = obs_flat_mat[obs_idx]
            mat_inv_buf[dst] = obs_flat_mat_inv[obs_idx]
            handle_buf[dst] = torch.tensor(obs_flat_handles[obs_idx], dtype=torch.int64)
            if obs_idx in art_flat:
                art_flat[obs_idx]["slots"] = dst.to(torch.int32)
                art_flat[obs_idx]["env_ids"] = collider_rows.to(torch.int32)

        # Bulk upload to GPU + zero-copy wrap + pack kernel
        self._coll_pos_t = pos_buf.contiguous().to(device)
        self._coll_mat_t = mat_buf.contiguous().to(device)
        self._coll_mat_inv_t = mat_inv_buf.contiguous().to(device)
        self._handles_flat = handle_buf.contiguous().to(device)

        self.wp_colliders = wp.zeros(total_slots, dtype=ColliderTransform, device=device)
        wp.launch(
            pack_collider_transforms,
            dim=total_slots,
            inputs=[
                wp.from_torch(self._coll_pos_t, dtype=wp.vec3),
                wp.from_torch(self._coll_mat_t, dtype=wp.mat33),
                wp.from_torch(self._coll_mat_inv_t, dtype=wp.mat33),
            ],
            outputs=[self.wp_colliders],
            device=device,
        )
        mesh_bounds = [
            torch.stack((wp.to_torch(mesh.points).amin(dim=0), wp.to_torch(mesh.points).amax(dim=0)))
            for mesh in meshes.values()
        ]
        bounds = torch.stack([torch.zeros((2, 3), device=device), *mesh_bounds])
        # Expand outward to absorb float32 transform/distance roundoff; the zero row stays disabled.
        padding = 8.0 * torch.finfo(bounds.dtype).eps * bounds.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        bounds[1:] += bounds.new_tensor([[[-1.0], [1.0]]]) * padding[1:]
        bounds_ids = {handle: i + 1 for i, handle in enumerate(meshes)}
        bounds_indices = torch.tensor(
            [bounds_ids.get(handle, 0) for handle in handle_buf.tolist()], dtype=torch.int32, device=device
        )
        wp.launch(
            _pack_collider_bounds,
            dim=total_slots,
            inputs=[wp.from_torch(bounds, dtype=wp.vec3), wp.from_torch(bounds_indices, dtype=wp.int32)],
            outputs=[self.wp_colliders],
            device=device,
        )

        # Convert remaining data to persistent warp arrays
        self.wp_local_pts = wp.from_torch(self.local_pts, dtype=wp.vec3)
        self.wp_body_ids = wp.from_torch(self._body_ids_i32, dtype=wp.int32)
        self.wp_handles = wp.from_torch(self._handles_flat, dtype=wp.uint64)
        self._prim_counts_flat = torch.cat(obstacle_prim_counts).contiguous().to(device)
        self.wp_prim_counts = wp.from_torch(self._prim_counts_flat, dtype=wp.int32)
        self._obstacle_row_offsets = row_offsets.to(device=device, dtype=torch.int32)
        self.wp_obstacle_row_offsets = wp.from_torch(self._obstacle_row_offsets, dtype=wp.int32)
        self._collider_rows = torch.zeros((num_obstacles, num_envs), dtype=torch.int32, device=device)
        all_env_rows = torch.arange(num_envs, dtype=torch.int32, device=device)
        for i, row_count in enumerate(obstacle_row_counts):
            if row_count == num_envs:
                self._collider_rows[i].copy_(all_env_rows)
        self.wp_collider_rows = wp.from_torch(self._collider_rows, dtype=wp.int32)
        self._variant_obstacles = variant_obstacles

        self._obs_root_scales = torch.cat(obstacle_root_scales).contiguous()
        self.wp_obs_root_scale = wp.from_torch(self._obs_root_scales, dtype=wp.vec3)
        # Articulated obstacles skip the per-call root update and keep identity rows
        # (their collider slots hold world transforms), so initialize quats to identity.
        self._obs_root_pos_t = torch.zeros((num_obstacles, num_envs, 3), device=device)
        self._obs_root_quat_t = torch.zeros((num_obstacles, num_envs, 4), device=device)
        self._obs_root_quat_t[..., 3] = 1.0
        self.wp_obs_root_pos = wp.from_torch(self._obs_root_pos_t, dtype=wp.vec3)
        self.wp_obs_root_quat = wp.from_torch(self._obs_root_quat_t, dtype=wp.quat)

        # Persistent GPU data for the per-call articulated collider refresh.
        self._articulated: dict[int, dict] = {}
        for obs_idx, flat in art_flat.items():
            pack = {
                "n": int(flat["env_ids"].numel()),
                "_env_ids": flat["env_ids"].contiguous().to(device),
                "_body_ids": flat["body_ids"].contiguous().to(device),
                "_slots": flat["slots"].contiguous().to(device),
                "_rel_pos": flat["rel_pos"].contiguous().to(device),
                "_rel_mat": flat["rel_mat"].contiguous().to(device),
            }
            pack["wp_env_ids"] = wp.from_torch(pack["_env_ids"], dtype=wp.int32)
            pack["wp_body_ids"] = wp.from_torch(pack["_body_ids"], dtype=wp.int32)
            pack["wp_slots"] = wp.from_torch(pack["_slots"], dtype=wp.int32)
            pack["wp_rel_pos"] = wp.from_torch(pack["_rel_pos"], dtype=wp.vec3)
            pack["wp_rel_mat"] = wp.from_torch(pack["_rel_mat"], dtype=wp.mat33)
            self._articulated[obs_idx] = pack

        # Grow to the largest queried subset, not every environment's full surface budget.
        self.wp_signs = wp.empty(0, dtype=float, device=device)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> torch.Tensor:
        """Check the selected environments' current source poses against their live obstacles."""
        if isinstance(env_ids, slice):
            env_ids = env.scene._ALL_INDICES[env_ids]
        ids = wp.from_torch(env_ids.to(device=env.device, dtype=torch.int32), dtype=wp.int32)
        rows = self.asset.mesh_variant_ids.warp if self._asset_uses_variants else self.wp_asset_rows
        return self._query(
            env, ids, ids, self.asset.data.body_link_pos_w.warp, self.asset.data.body_link_quat_w.warp, rows
        )

    def check_poses(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        body_poses_w: torch.Tensor,
        variant_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Check hypothetical source poses against live obstacles without changing scene state.

        Args:
            env: Environment owning the configured source and obstacles.
            env_ids: Live obstacle environment indices, shape ``(N,)``. Repeated indices are supported.
            body_poses_w: Source body positions [m] and unit quaternions (x, y, z, w), shape
                ``(N, len(asset.body_names), 7)``, in the source asset's body order.
            variant_ids: Source mesh-variant indices, shape ``(N,)``. Defaults to the live variants
                selected by ``env_ids``, or zero for a source without variants.

        Returns:
            Collision-free flags, shape ``(N,)``, using the same geometry and thresholds as :meth:`__call__`.
        """
        count = len(env_ids)
        if env_ids.ndim != 1 or body_poses_w.shape != (count, len(self.asset.body_names), 7):
            raise ValueError("Expected environment indices (N,) and source body poses (N, num_asset_bodies, 7).")
        if env_ids.dtype not in (torch.int32, torch.int64) or ((env_ids < 0) | (env_ids >= env.num_envs)).any():
            raise ValueError("Expected integer environment indices in [0, env.num_envs).")
        poses = body_poses_w.to(device=env.device, dtype=torch.float32)
        if variant_ids is None:
            variant_ids = (
                self.asset.mesh_variant_ids.torch[env_ids]
                if self._asset_uses_variants
                else torch.zeros(count, device=env.device, dtype=torch.int32)
            )
        if variant_ids.shape != (count,) or variant_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer source variant indices with shape (N,).")
        if ((variant_ids < 0) | (variant_ids >= self.local_pts.shape[0])).any():
            raise ValueError("Expected source variant indices within the configured source geometry.")
        ids = wp.from_torch(env_ids.to(device=env.device, dtype=torch.int32), dtype=wp.int32)
        sources = wp.from_torch(torch.arange(count, device=env.device, dtype=torch.int32), dtype=wp.int32)
        rows = wp.from_torch(variant_ids.to(device=env.device, dtype=torch.int32), dtype=wp.int32)
        valid = self._query(
            env,
            ids,
            sources,
            wp.from_torch(poses[..., :3], dtype=wp.vec3),
            wp.from_torch(poses[..., 3:], dtype=wp.quat),
            rows,
        )
        return valid & torch.isfinite(poses).all(dim=(1, 2))

    def _query(
        self,
        env: ManagerBasedRLEnv,
        env_ids: wp.array,
        source_ids: wp.array,
        body_pos_w: wp.array,
        body_quat_w: wp.array,
        asset_rows: wp.array,
    ) -> torch.Tensor:
        num_query_envs = env_ids.shape[0]
        num_bodies = len(self.body_ids)
        num_points = self.cfg.num_points
        if num_query_envs == 0:
            return torch.empty(0, device=env.device, dtype=torch.bool)

        for i, obs in enumerate(self.obstacles):
            if i in self._variant_obstacles:
                self._collider_rows[i].copy_(obs.mesh_variant_ids.torch)
            art = self._articulated.get(i)
            if art is not None:
                wp.launch(
                    update_articulated_collider_transforms,
                    dim=art["n"],
                    inputs=[
                        obs.data.body_link_pos_w.warp,
                        obs.data.body_link_quat_w.warp,
                        art["wp_env_ids"],
                        art["wp_body_ids"],
                        art["wp_slots"],
                        art["wp_rel_pos"],
                        art["wp_rel_mat"],
                    ],
                    outputs=[self.wp_colliders],
                    device=env.device,
                )
            else:
                wp.launch(
                    update_root_poses,
                    dim=env.num_envs,
                    inputs=[
                        self.wp_obs_root_pos,
                        self.wp_obs_root_quat,
                        obs.data.root_pos_w.warp,
                        obs.data.root_quat_w.warp,
                        i,
                    ],
                    device=env.device,
                )

        total_threads = num_query_envs * num_bodies * num_points
        if self.wp_signs.size < total_threads:
            self.wp_signs = wp.empty(total_threads, dtype=float, device=env.device)
        wp.launch(
            get_signed_distance_mega,
            dim=total_threads,
            inputs=[
                body_pos_w,
                body_quat_w,
                self.wp_body_ids,
                asset_rows,
                self.wp_local_pts,
                env_ids,
                source_ids,
                self.wp_obs_root_pos,
                self.wp_obs_root_quat,
                self.wp_obs_root_scale,
                self.wp_colliders,
                self.wp_handles,
                self.wp_prim_counts,
                self.wp_obstacle_row_offsets,
                self.wp_collider_rows,
                self.wp_min_dist_offsets,
                float(self.cfg.max_dist),
                float(self.cfg.min_dist),
                self._check_dist,
                num_bodies,
                num_points,
                self.num_obstacles,
                self.max_prims,
            ],
            outputs=[self.wp_signs],
            device=env.device,
        )

        signs = wp.to_torch(self.wp_signs)[:total_threads].view(num_query_envs, num_bodies, num_points)
        return signs.amin(dim=(1, 2)) >= self.cfg.min_dist
