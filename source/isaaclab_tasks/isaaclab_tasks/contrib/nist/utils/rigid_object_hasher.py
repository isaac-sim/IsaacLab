# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib

import numpy as np
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab import cloner
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import get_current_stage
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source


class RigidObjectHasher:
    """Read collider hashes and root-relative transforms from the current USD stage."""

    def __init__(self, num_envs: int, prim_path_pattern: str):
        self.num_root = 0
        self.collider_prims: list[Usd.Prim] = []
        self.collider_prim_hashes = torch.empty(0, dtype=torch.int64)
        self.collider_prim_env_ids = torch.empty(0, dtype=torch.int64)
        self.collider_rel_pos = torch.empty(0, 3)
        self.collider_rel_mat = torch.empty(0, 3, 3)
        self.collider_rel_mat_inv = torch.empty(0, 3, 3)
        self.root_prim_hashes = torch.empty(0, dtype=torch.int64)
        self.root_prim_scales = torch.empty(0, 3)

        xform_cache = UsdGeom.XformCache()
        stage = get_current_stage()

        def is_collider(p) -> bool:
            return p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone") and p.HasAPI(
                UsdPhysics.CollisionAPI
            )

        # Discover collider sources from the clone plan rather than walking every cloned env
        # subtree on stage. Each plan row is one source variant; ``env_ids`` are exactly the
        # envs its clone-mask populates. This is correct for heterogeneous plans (a clone_mask
        # with mixed True/False across variant rows -> different geometry per env) and reduces
        # to a single source covering all envs for homogeneous scenes. Walking the clone source
        # (instead of the cloned subtrees) also works when there is no USD cloning, where only
        # the source instances exist on stage and the per-env layout comes from the clone plan.
        plan = SimulationContext.instance().get_clone_plan()
        source_rows: list[tuple[str, tuple[int, ...]]] = []
        if plan is not None:
            source_rows = [
                (source_path, env_ids)
                for _src_root, _dst_tmpl, source_path, env_ids in cloner.query.iter_sources(plan, prim_path_pattern)
            ]
        if not source_rows:
            # No clone plan (or pattern not owned by it): resolve a single source instance and
            # treat every env as a clone of it.
            fallback = resolve_matching_prims_from_source(prim_path_pattern, raise_if_no_matches=False)
            if not fallback:
                return
            source_rows = [(fallback[0][0].GetPath().pathString, tuple(range(num_envs)))]

        num_roots = num_envs
        collider_prims: list[Usd.Prim] = []
        collider_prim_env_ids: list[int] = []
        collider_prim_hashes: list[int] = []
        collider_rel_pos_list: list[torch.Tensor] = []
        collider_rel_mat_list: list[torch.Tensor] = []
        collider_rel_mat_inv_list: list[torch.Tensor] = []
        # Indexed by env id so downstream root-indexed reads stay aligned regardless of row order.
        root_prim_hashes: list[int] = [0] * num_roots
        root_prim_scales: list = [None] * num_roots

        for source_path, env_ids in source_rows:
            asset_prim = stage.GetPrimAtPath(source_path)
            # ``traverse_instance_prims=True`` so colliders authored inside instanceable asset
            # references are still discovered (the hasher needs the actual mesh geometry).
            coll_prims = get_all_matching_child_prims(source_path, predicate=is_collider, traverse_instance_prims=True)
            if len(coll_prims) == 0:
                return

            # Hash this source variant's colliders once and capture root-relative transforms.
            root_xf = xform_cache.GetLocalToWorldTransform(asset_prim)
            root_scale = torch.tensor(Gf.Transform(root_xf).GetScale())
            root_hash = hashlib.sha256()
            src_collider_hashes: list[int] = []
            src_rel_pos: list[torch.Tensor] = []
            src_rel_mat: list[torch.Tensor] = []
            src_rel_mat_inv: list[torch.Tensor] = []
            for prim in coll_prims:
                child_xf = xform_cache.GetLocalToWorldTransform(prim)
                rel_mat4 = np.array(child_xf * root_xf.GetInverse(), dtype=np.float64)
                rel_mat3 = torch.tensor(rel_mat4[:3, :3].T, dtype=torch.float32)
                rel_pos = torch.tensor(rel_mat4[3, :3], dtype=torch.float32)
                rel_mat3_inv = torch.linalg.inv(rel_mat3.to(torch.float64)).to(torch.float32)

                src_rel_pos.append(rel_pos)
                src_rel_mat.append(rel_mat3)
                src_rel_mat_inv.append(rel_mat3_inv)

                h = hashlib.sha256()
                h.update(np.round(rel_mat4[:3, :3] * 50).astype(np.int64))
                h.update(np.round(rel_mat4[3, :3] * 50).astype(np.int64))
                prim_type = prim.GetTypeName()
                h.update(prim_type.encode("utf-8"))
                if prim_type == "Mesh":
                    verts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32)
                    h.update(verts.tobytes())
                elif prim_type == "Cube":
                    h.update(np.float32(UsdGeom.Cube(prim).GetSizeAttr().Get()).tobytes())
                elif prim_type == "Sphere":
                    h.update(np.float32(UsdGeom.Sphere(prim).GetRadiusAttr().Get()).tobytes())
                elif prim_type == "Cylinder":
                    c = UsdGeom.Cylinder(prim)
                    h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Capsule":
                    c = UsdGeom.Capsule(prim)
                    h.update(c.GetAxisAttr().Get().encode("utf-8"))
                    h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Cone":
                    c = UsdGeom.Cone(prim)
                    h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                collider_hash = h.digest()
                root_hash.update(collider_hash)
                src_collider_hashes.append(int.from_bytes(collider_hash[:8], "little", signed=True))

            root_hash_int = int.from_bytes(root_hash.digest()[:8], "little", signed=True)

            # Apply this variant to every env its clone-mask row populates.
            for e in env_ids:
                collider_prims.extend(coll_prims)
                collider_prim_env_ids.extend([e] * len(coll_prims))
                collider_prim_hashes.extend(src_collider_hashes)
                collider_rel_pos_list.extend(src_rel_pos)
                collider_rel_mat_list.extend(src_rel_mat)
                collider_rel_mat_inv_list.extend(src_rel_mat_inv)
                root_prim_hashes[e] = root_hash_int
                root_prim_scales[e] = root_scale

        self.num_root = num_roots
        self.collider_prims = collider_prims
        self.collider_prim_hashes = torch.tensor(collider_prim_hashes, dtype=torch.int64)
        self.collider_prim_env_ids = torch.tensor(collider_prim_env_ids, dtype=torch.int64)
        self.collider_rel_pos = torch.stack(collider_rel_pos_list) if collider_rel_pos_list else torch.empty(0, 3)
        self.collider_rel_mat = torch.stack(collider_rel_mat_list) if collider_rel_mat_list else torch.empty(0, 3, 3)
        self.collider_rel_mat_inv = (
            torch.stack(collider_rel_mat_inv_list) if collider_rel_mat_inv_list else torch.empty(0, 3, 3)
        )
        self.root_prim_hashes = torch.tensor(root_prim_hashes, dtype=torch.int64)
        self.root_prim_scales = torch.stack(root_prim_scales)
