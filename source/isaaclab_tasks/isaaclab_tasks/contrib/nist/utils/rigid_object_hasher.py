# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab import cloner
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import get_current_stage
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source
from isaaclab.utils.string import resolve_matching_names


class RigidObjectHasher:
    """Compute geometry hashes and transforms, with optional per-body visual fallback.

    ``body_names`` scopes geometry to its nearest rigid-body owner using resolved, exact names.
    ``authored_collision`` preserves observation geometry: all authored colliders in traversal order.
    ``collision`` selects enabled colliders; ``collision_or_visual`` additionally uses visible meshes
    for bodies without enabled colliders. Reset safety checks select these policies explicitly.
    An explicit ``stage`` reads geometry independently of the simulation and requires ``source_paths``.
    """

    def __init__(
        self,
        num_envs: int,
        prim_path_pattern: str,
        device: str | torch.device = "cpu",
        rigid_body_root: bool = False,
        compact_sources: bool = False,
        source_paths: Sequence[str] | None = None,
        geometry_policy: Literal["authored_collision", "collision", "collision_or_visual"] = "authored_collision",
        body_names: Sequence[str] | None = None,
        stage: Usd.Stage | None = None,
    ):
        self.prim_path_pattern = prim_path_pattern
        self.device = device
        sources = tuple(source_paths) if source_paths is not None else None
        if (stage is not None or sources is not None) and (sources is None or not compact_sources):
            raise ValueError("Explicit geometry sources require source_paths and compact_sources=True.")
        body_names = tuple(sorted(set(body_names))) if body_names is not None else None
        if geometry_policy not in ("authored_collision", "collision", "collision_or_visual"):
            raise ValueError(f"Unknown collision geometry policy: {geometry_policy!r}.")
        self.num_root = 0
        self.collider_prims: list[Usd.Prim] = []
        self.collider_prim_hashes = torch.empty(0, dtype=torch.int64, device=device)
        self.collider_prim_env_ids = torch.empty(0, dtype=torch.int64, device=device)
        self.collider_prim_group_ids = torch.empty(0, dtype=torch.int64, device=device)
        self.collider_rel_pos = torch.empty(0, 3, device=device)
        self.collider_rel_mat = torch.empty(0, 3, 3, device=device)
        self.collider_rel_mat_inv = torch.empty(0, 3, 3, device=device)
        self.root_prim_hashes = torch.empty(0, dtype=torch.int64, device=device)
        self.root_prim_scales = torch.empty(0, 3, device=device)
        self.point_cloud_cache: dict[tuple, torch.Tensor] = {}

        xform_cache = UsdGeom.XformCache()
        stage = get_current_stage() if stage is None else stage

        source_rows = []
        if sources is not None:
            source_rows = [(path, ()) for path in sources]
        else:
            # Resolve authored sources through native world-prototype membership.
            plan = SimulationContext.instance().get_clone_plan()
            matches = []
            if plan is not None:
                prototype_paths = cloner.path.get_asset_prototype_paths(plan)
                templates, starts, worlds, world_starts = cloner.path.get_world_prototype_asset_templates(
                    plan, include_world_indices=True
                )
                for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
                    for index in range(*starts[group : group + 2]):
                        if (matched := cloner.path.match(prim_path_pattern, templates[index])) is not None:
                            matches.append((group, index, matched))
            if matches:
                suffix = min((matched.suffix for _, _, matched in matches), key=len)
                source_worlds = {}
                for group, index, matched in matches:
                    if matched.suffix != suffix:
                        continue
                    env_ids = worlds[world_starts[group] : world_starts[group + 1]]
                    selected, _ = resolve_matching_names(
                        matched.instance, env_ids.astype(str), raise_when_no_match=False
                    )
                    if selected:
                        path = prototype_paths[plan.topology.world_prototypes[index]] + suffix
                        source_worlds.setdefault(path, []).extend(env_ids[selected])
                source_rows = list(source_worlds.items())
            else:
                fallback = resolve_matching_prims_from_source(prim_path_pattern, raise_if_no_matches=False)
                if fallback:
                    source_rows = [(fallback[0][0].GetPath().pathString, range(num_envs))]
        if not source_rows:
            return

        num_roots = len(source_rows) if compact_sources else num_envs
        collider_prims: list[Usd.Prim] = []
        collider_prim_env_ids: list[int] = []
        collider_prim_group_ids: list[int] = []
        collider_prim_hashes: list[int] = []
        collider_rel_pos_list: list[torch.Tensor] = []
        collider_rel_mat_list: list[torch.Tensor] = []
        collider_rel_mat_inv_list: list[torch.Tensor] = []
        root_prim_hashes: list[int] = [0] * num_roots
        root_prim_scales = [torch.ones(3)] * num_roots

        for variant_id, (source_path, env_ids) in enumerate(source_rows):
            asset_prim = stage.GetPrimAtPath(source_path)
            if rigid_body_root:
                rigid_bodies = get_all_matching_child_prims(
                    source_path,
                    predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    traverse_instance_prims=True,
                    stage=stage,
                )
                if len(rigid_bodies) != 1:
                    raise ValueError(f"Expected one rigid body below '{source_path}', found {len(rigid_bodies)}.")
                asset_prim = rigid_bodies[0]
                source_path = asset_prim.GetPath().pathString
            geometry_prims = get_all_matching_child_prims(
                source_path,
                predicate=lambda prim: prim.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"),
                traverse_instance_prims=True,
                stage=stage,
            )
            geometry_by_body: dict[str, list[Usd.Prim]] = {}
            for prim in geometry_prims:
                body = prim
                while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if body_names is not None and (not body.IsValid() or body.GetName() not in body_names):
                    continue
                owner_path = (
                    str(body.GetPath()) if body.IsValid() and geometry_policy != "authored_collision" else source_path
                )
                geometry_by_body.setdefault(owner_path, []).append(prim)
            coll_prims = []
            src_group_ids = []
            for body_prims in geometry_by_body.values():
                selected = [
                    prim
                    for prim in body_prims
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                    and (
                        geometry_policy == "authored_collision"
                        or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    )
                ]
                if not selected and geometry_policy == "collision_or_visual":
                    selected = [
                        prim
                        for prim in body_prims
                        if prim.GetTypeName() == "Mesh"
                        and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
                        and UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
                    ]
                    src_group_ids.extend([len(coll_prims)] * len(selected))
                else:
                    src_group_ids.extend(range(len(coll_prims), len(coll_prims) + len(selected)))
                coll_prims.extend(selected)

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
                h.update(rel_mat4.tobytes())
                prim_type = prim.GetTypeName()
                h.update(prim_type.encode("utf-8"))
                if prim_type == "Mesh":
                    verts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32)
                    h.update(verts.tobytes())
                    h.update(np.asarray(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(), dtype=np.int32).tobytes())
                    h.update(np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes())
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

            root_ids = (variant_id,) if compact_sources else env_ids
            for e in root_ids:
                collider_prims.extend(coll_prims)
                collider_prim_env_ids.extend([e] * len(coll_prims))
                collider_prim_group_ids.extend(src_group_ids)
                collider_prim_hashes.extend(src_collider_hashes)
                collider_rel_pos_list.extend(src_rel_pos)
                collider_rel_mat_list.extend(src_rel_mat)
                collider_rel_mat_inv_list.extend(src_rel_mat_inv)
                root_prim_hashes[e] = root_hash_int
                root_prim_scales[e] = root_scale

        self.num_root = num_roots
        self.collider_prims = collider_prims
        self.collider_prim_hashes = torch.tensor(collider_prim_hashes, dtype=torch.int64, device=device)
        self.collider_prim_env_ids = torch.tensor(collider_prim_env_ids, dtype=torch.int64, device=device)
        self.collider_prim_group_ids = torch.tensor(collider_prim_group_ids, dtype=torch.int64, device=device)
        self.collider_rel_pos = (
            torch.stack(collider_rel_pos_list).to(device) if collider_rel_pos_list else torch.empty(0, 3, device=device)
        )
        self.collider_rel_mat = (
            torch.stack(collider_rel_mat_list).to(device)
            if collider_rel_mat_list
            else torch.empty(0, 3, 3, device=device)
        )
        self.collider_rel_mat_inv = (
            torch.stack(collider_rel_mat_inv_list).to(device)
            if collider_rel_mat_inv_list
            else torch.empty(0, 3, 3, device=device)
        )
        self.root_prim_hashes = torch.tensor(root_prim_hashes, dtype=torch.int64, device=device)
        self.root_prim_scales = torch.stack(root_prim_scales).to(device)
