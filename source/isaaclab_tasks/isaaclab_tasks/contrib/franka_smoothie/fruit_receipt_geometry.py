# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-tolerance receipt from enabled authored fruit colliders and actual cup planes."""

from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull

from pxr import Usd, UsdGeom, UsdPhysics

ASSETS = Path(__file__).resolve().parent / "assets"
KINDS = ("strawberry", "blueberry", "blackberry", "mango")


def enabled(prim):
    """Return whether the authored collision API is enabled."""
    return (
        prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    )


def transformed_points(prim, cache, body_inverse):
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * body_inverse)
    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    return (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]


@lru_cache(maxsize=1)
def load_geometry():
    """Load convex support vertices [m] and cup inward containment bounds [m]."""
    stage = Usd.Stage.Open(str(ASSETS / "cup.usda"))
    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath("/Asset/Cup")
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    normals, distances, rims = [], [], []
    for prim in stage.GetPrimAtPath("/Asset/Cup/Wall").GetChildren():
        assert enabled(prim) and prim.IsA(UsdGeom.Mesh)
        points = transformed_points(prim, cache, inverse)
        radii = np.linalg.norm(points[:, :2], axis=-1)
        inner = np.unique(points[np.abs(radii - radii.min()) < 1e-7, :2], axis=0)
        assert inner.shape == (2, 2)
        edge = inner[1] - inner[0]
        normal = np.array([edge[1], -edge[0]])
        normal /= np.linalg.norm(normal)
        if normal @ inner.mean(0) < 0:
            normal *= -1
        distance = float(inner[0] @ normal)
        assert np.max(np.abs(inner @ normal - distance)) < 1e-14
        normals.append(normal)
        distances.append(distance)
        rims.append(float(points[:, 2].max()))
    assert len(normals) == 32 and max(rims) - min(rims) < 1e-9
    floor_prim = stage.GetPrimAtPath("/Asset/Cup/Floor")
    assert enabled(floor_prim)
    floor = UsdGeom.Cylinder(floor_prim)
    matrix = np.asarray(cache.GetLocalToWorldTransform(floor_prim) * inverse)
    assert str(floor.GetAxisAttr().Get()) == "Z" and np.allclose(matrix[:3, :3], np.eye(3))
    floor_top = float(matrix[3, 2] + floor.GetHeightAttr().Get() / 2)
    geometry, descriptions = {}, {}
    for kind in KINDS:
        path = ASSETS / f"{kind}.{'usda' if kind == 'mango' else 'usdc'}"
        fruit = Usd.Stage.Open(str(path))
        cache = UsdGeom.XformCache()
        roots = [p for p in fruit.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert len(roots) == 1
        inverse = cache.GetLocalToWorldTransform(roots[0]).GetInverse()
        vertices, parts = [], []
        for prim in fruit.Traverse():
            if not enabled(prim):
                continue
            assert prim.IsA(UsdGeom.Mesh)
            assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
            points = transformed_points(prim, cache, inverse)
            vertices.append(points)
            parts.append({"path": str(prim.GetPath()), "point_count": len(points)})
        all_points = np.concatenate(vertices)
        hull = ConvexHull(all_points)
        support = np.ascontiguousarray(all_points[hull.vertices], dtype=np.float64)
        geometry[kind] = support
        descriptions[kind] = {
            "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "enabled_colliders": parts,
            "authored_point_count": len(all_points),
            "support_point_count": len(support),
            "support_vertices_float64_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
        }
    metadata = {
        "schema": "authored_fruit_hull_receipt",
        "version": 1,
        "tolerance_m": 0.0,
        "method": (
            "Every enabled authored convexHull vertex is inside all32 actual inner-wall planes and floor/rim "
            "planes. Outer support vertices preserve exact maxima in this convex receiving region."
        ),
        "floating_point": "Normalized recorded XYZW quaternions, float64 transforms and plane comparisons <=0.",
        "cup_sha256": hashlib.sha256((ASSETS / "cup.usda").read_bytes()).hexdigest(),
        "inner_wall_normals_xy": np.asarray(normals).tolist(),
        "inner_wall_distances_m": distances,
        "floor_top_m": floor_top,
        "rim_m": min(rims),
        "fruit_geometry": descriptions,
        "legacy_center_metric": "Recorded separately; does not advance this version3 receipt gate.",
    }
    return geometry, np.asarray(normals), np.asarray(distances), floor_top, min(rims), metadata


def geometry_contract():
    """Return the serialized authoritative geometry identity without sharing mutable configuration."""
    return copy.deepcopy(load_geometry()[-1])


def rotation_matrix(q):
    """Return rotation matrices from finite nonzero XYZW quaternions."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-30)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class WholeHullReceipt:
    """Measure each fruit's whole-collider containment in the actual receiving cup."""

    def __init__(self, names: tuple[str, ...], device: str):
        geometry, normals, distances, self.floor, self.rim, _ = load_geometry()
        assert len(names) == 16 and set(n.split("_")[0] for n in names) == set(KINDS)
        self.indices = {
            kind: torch.tensor([i for i, n in enumerate(names) if n.split("_")[0] == kind], device=device)
            for kind in KINDS
        }
        self.vertices = {
            kind: torch.tensor(value, dtype=torch.float64, device=device) for kind, value in geometry.items()
        }
        self.normals = torch.tensor(normals, dtype=torch.float64, device=device)
        self.distances = torch.tensor(distances, dtype=torch.float64, device=device)

    def measure(self, cup_pose: torch.Tensor, fruit_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contained masks [N,16] and maximum plane violations [m] from measured world poses."""
        cup = cup_pose.to(torch.float64)
        fruit = fruit_poses.to(torch.float64)
        valid_cup = torch.isfinite(cup).all(-1) & (torch.linalg.vector_norm(cup[:, 3:], dim=-1) > 1e-12)
        valid = (
            torch.isfinite(fruit).all(-1)
            & (torch.linalg.vector_norm(fruit[:, :, 3:], dim=-1) > 1e-12)
            & valid_cup[:, None]
        )
        cup_r = rotation_matrix(cup[:, 3:]).transpose(-1, -2)
        fruit_r = rotation_matrix(fruit[:, :, 3:])
        relative_r = torch.einsum("nij,nfjk->nfik", cup_r, fruit_r)
        center = torch.einsum("nij,nfj->nfi", cup_r, fruit[:, :, :3] - cup[:, None, :3])
        violations = torch.empty(fruit.shape[:2], dtype=torch.float64, device=fruit.device)
        for kind in KINDS:
            ids = self.indices[kind]
            points = torch.einsum("nfij,vj->nfvi", relative_r[:, ids], self.vertices[kind]) + center[:, ids, None, :]
            sides = (points[:, :, :, :2] @ self.normals.T - self.distances).amax(dim=(-1, -2))
            low = self.floor - points[:, :, :, 2].amin(-1)
            high = points[:, :, :, 2].amax(-1) - self.rim
            violations[:, ids] = torch.maximum(torch.maximum(sides, low), high)
        violations = torch.where(valid, violations, torch.full_like(violations, float("inf")))
        return violations <= 0.0, violations

    def all_types(self, inside: torch.Tensor) -> torch.Tensor:
        """Require at least one completely contained fruit of every authored kind."""
        return torch.stack([inside[:, ids].any(-1) for ids in self.indices.values()], dim=-1).all(-1)
