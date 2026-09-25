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

from isaaclab.sim.utils import resolve_matching_prims_from_source

from isaaclab_tasks.contrib.nist.utils import mesh_ops as _mesh_ops
from isaaclab_tasks.contrib.nist.utils.rigid_object_hasher import RigidObjectHasher
from isaaclab_tasks.contrib.nist.utils.sdf import (
    ColliderTransform,
    get_signed_distance_mega,
    pack_collider_transforms,
    update_articulated_collider_transforms,
    update_root_poses,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.contrib.nist.utils.collision_analyzer_cfg import CollisionAnalyzerCfg


class CollisionAnalyzer:
    cfg: CollisionAnalyzerCfg

    def __init__(self, cfg: CollisionAnalyzerCfg, env: ManagerBasedRLEnv):
        from pxr import UsdGeom, UsdPhysics

        self.cfg = cfg
        self.asset: RigidObject = env.scene[cfg.asset_cfg.name]
        self.obstacles: list[RigidObject] = [env.scene[cfg.name] for cfg in cfg.obstacle_cfgs]
        device = env.device
        body_names = (
            self.asset.body_names
            if cfg.asset_cfg.body_names is None
            else [self.asset.body_names[i] for i in cfg.asset_cfg.body_ids]
        )
        if isinstance(body_names, str):
            body_names = [body_names]

        self.body_ids = []
        self.local_pts = []
        for body_name in body_names:
            _, prim_path_pattern = resolve_matching_prims_from_source(
                self.asset.cfg.prim_path,
                predicate=lambda p: p.GetName() == body_name and p.HasAPI(UsdPhysics.RigidBodyAPI),
                expected_num_matches=1,
            )[0]
            local_pts = _mesh_ops.sample_object_point_cloud(
                num_envs=env.num_envs,
                num_points=cfg.num_points,
                prim_path_pattern=prim_path_pattern,
                device=device,
            )
            if local_pts is not None:
                self.local_pts.append(local_pts.view(env.num_envs, 1, cfg.num_points, 3))
                self.body_ids.append(self.asset.body_names.index(body_name))
        self.local_pts = torch.cat(self.local_pts, dim=1).contiguous()
        self._body_ids_i32 = torch.tensor(self.body_ids, dtype=torch.int32, device=device)

        num_obstacles = len(self.obstacles)
        num_envs = env.num_envs

        num_coll_per_obs_env = torch.empty((num_obstacles, num_envs), dtype=torch.int32, device=device)
        obstacle_root_scales = torch.empty((num_obstacles, num_envs, 3), dtype=torch.float, device=device)

        # First pass: mesh dedup + collect flat CPU data per obstacle.
        # We store flat arrays and env_ids from the hasher, then scatter
        # into the strided layout after max_prims is known.
        obs_flat_pos: list[torch.Tensor] = []
        obs_flat_mat: list[torch.Tensor] = []
        obs_flat_mat_inv: list[torch.Tensor] = []
        obs_flat_env_ids: list[torch.Tensor] = []
        obs_flat_handles: list[list[int]] = []

        # Articulated obstacles (multiple body links): the hasher's root-relative
        # transforms freeze the spawn articulation, so also capture each collider's
        # owning body link and its static BODY-relative transform. The runtime pass
        # recomposes world transforms from live body poses instead of the root pose.
        art_flat: dict[int, dict[str, torch.Tensor]] = {}
        self._meshes: dict[int, wp.Mesh] = {}

        for i, obstacle in enumerate(self.obstacles):
            start = time.perf_counter()
            obs_h = RigidObjectHasher(num_envs, prim_path_pattern=obstacle.cfg.prim_path)

            rel_pos_cpu = obs_h.collider_rel_pos
            rel_mat_cpu = obs_h.collider_rel_mat
            rel_mat_inv_cpu = obs_h.collider_rel_mat_inv
            env_ids_cpu = obs_h.collider_prim_env_ids
            hashes_cpu = obs_h.collider_prim_hashes

            handles_flat: list[int] = []
            for j, prim in enumerate(obs_h.collider_prims):
                p_hash = hashes_cpu[j].item()
                if p_hash in self._meshes:
                    handles_flat.append(self._meshes[p_hash].id)
                else:
                    wp_mesh = _mesh_ops.prim_to_warp_mesh(prim, device=device, relative_to_world=False)
                    self._meshes[p_hash] = wp_mesh
                    handles_flat.append(wp_mesh.id)

            obs_flat_pos.append(rel_pos_cpu)
            obs_flat_mat.append(rel_mat_cpu)
            obs_flat_mat_inv.append(rel_mat_inv_cpu)
            obs_flat_env_ids.append(env_ids_cpu)
            obs_flat_handles.append(handles_flat)

            num_coll_per_obs_env[i] = torch.bincount(obs_h.collider_prim_env_ids, minlength=num_envs).to(device)
            obstacle_root_scales[i] = obs_h.root_prim_scales.to(device)

            if len(obstacle.body_names) > 1:
                xform_cache = UsdGeom.XformCache()
                per_prim: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
                body_ids_flat: list[int] = []
                rel_pos_body: list[torch.Tensor] = []
                rel_mat_body: list[torch.Tensor] = []
                for prim in obs_h.collider_prims:
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
                        rel4 = np.array(
                            xform_cache.GetLocalToWorldTransform(prim)
                            * xform_cache.GetLocalToWorldTransform(link).GetInverse(),
                            dtype=np.float64,
                        )
                        cached = (
                            obstacle.body_names.index(link.GetName()),
                            torch.tensor(rel4[3, :3], dtype=torch.float32),
                            torch.tensor(rel4[:3, :3].T, dtype=torch.float32),
                        )
                        per_prim[key] = cached
                    body_ids_flat.append(cached[0])
                    rel_pos_body.append(cached[1])
                    rel_mat_body.append(cached[2])
                art_flat[i] = {
                    "body_ids": torch.tensor(body_ids_flat, dtype=torch.int32),
                    "rel_pos": torch.stack(rel_pos_body),
                    "rel_mat": torch.stack(rel_mat_body),
                }
                # Body-relative transforms already carry the body->collider scale, so
                # the kernel-side root composition must stay identity (scale = 1).
                obstacle_root_scales[i] = 1.0

            pc_time = time.perf_counter() - start
            print(f"Sampled {len(obs_h.collider_prims)} wp meshes at '{obstacle.cfg.prim_path}' in {pc_time:.3f}s")

        max_prims = int(torch.max(num_coll_per_obs_env).item())
        self.max_prims = max_prims
        self.num_obstacles = num_obstacles

        # Second pass: scatter flat data into strided layout.
        # Kernel addresses colliders as: obs_idx * num_envs * max_prims + env_id * max_prims + prim_slot
        total_slots = num_obstacles * num_envs * max_prims
        pos_buf = torch.zeros(total_slots, 3)
        mat_buf = torch.zeros(total_slots, 3, 3)
        mat_inv_buf = torch.zeros(total_slots, 3, 3)
        handle_buf = torch.zeros(total_slots, dtype=torch.int64)

        for obs_idx in range(num_obstacles):
            collider_env_ids = obs_flat_env_ids[obs_idx]
            num_colliders = len(collider_env_ids)
            slot_offsets = torch.zeros(num_colliders, dtype=torch.long)
            placed_per_env = torch.zeros(num_envs, dtype=torch.long)
            for c in range(num_colliders):
                eid = collider_env_ids[c].item()
                slot_offsets[c] = placed_per_env[eid]
                placed_per_env[eid] += 1
            dst = obs_idx * num_envs * max_prims + collider_env_ids * max_prims + slot_offsets
            pos_buf[dst] = obs_flat_pos[obs_idx]
            mat_buf[dst] = obs_flat_mat[obs_idx]
            mat_inv_buf[dst] = obs_flat_mat_inv[obs_idx]
            handle_buf[dst] = torch.tensor(obs_flat_handles[obs_idx], dtype=torch.int64)
            if obs_idx in art_flat:
                art_flat[obs_idx]["slots"] = dst.to(torch.int32)
                art_flat[obs_idx]["env_ids"] = collider_env_ids.to(torch.int32)

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

        # Convert remaining data to persistent warp arrays
        self.wp_local_pts = wp.from_torch(self.local_pts, dtype=wp.vec3)
        self.wp_body_ids = wp.from_torch(self._body_ids_i32, dtype=wp.int32)
        self.wp_handles = wp.from_torch(self._handles_flat, dtype=wp.uint64)
        self._prim_counts_flat = num_coll_per_obs_env.reshape(-1).contiguous()
        self.wp_prim_counts = wp.from_torch(self._prim_counts_flat, dtype=wp.int32)

        self._obs_root_scales = obstacle_root_scales.contiguous()
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

        # Pre-allocate reusable buffers
        num_bodies = len(self.body_ids)
        num_points = cfg.num_points
        max_output = num_envs * num_bodies * num_points
        self.wp_signs = wp.zeros(max_output, dtype=float, device=device)
        self._env_ids_i32 = torch.empty(num_envs, dtype=torch.int32, device=device)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        num_query_envs = len(env_ids)
        num_bodies = len(self.body_ids)
        num_points = self.cfg.num_points

        for i, obs in enumerate(self.obstacles):
            art = self._articulated.get(i)
            if art is not None:
                wp.launch(
                    update_articulated_collider_transforms,
                    dim=art["n"],
                    inputs=[
                        obs.data.body_link_pos_w,
                        obs.data.body_link_quat_w,
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
                    inputs=[self.wp_obs_root_pos, self.wp_obs_root_quat, obs.data.root_pos_w, obs.data.root_quat_w, i],
                    device=env.device,
                )

        self._env_ids_i32[:num_query_envs] = env_ids[:num_query_envs].to(torch.int32)
        wp_env_ids = wp.from_torch(self._env_ids_i32[:num_query_envs], dtype=wp.int32)
        total_threads = num_query_envs * num_bodies * num_points
        wp.launch(
            get_signed_distance_mega,
            dim=total_threads,
            inputs=[
                self.asset.data.body_link_pos_w,
                self.asset.data.body_link_quat_w,
                self.wp_body_ids,
                self.wp_local_pts,
                wp_env_ids,
                self.wp_obs_root_pos,
                self.wp_obs_root_quat,
                self.wp_obs_root_scale,
                self.wp_colliders,
                self.wp_handles,
                self.wp_prim_counts,
                float(self.cfg.max_dist),
                self.cfg.min_dist != 0.0,
                env.num_envs,
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
