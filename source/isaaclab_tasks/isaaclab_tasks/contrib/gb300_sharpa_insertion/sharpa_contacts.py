# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-surface fingertip colliders authored only on the spawned Sharpa robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.utils.mesh import create_trimesh_from_geom_mesh

if TYPE_CHECKING:
    from pxr import Usd


@sim_utils.clone
def spawn_sharpa_contact_meshes(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the native robot and match its fingertip collision surfaces to its visuals.

    Positions remain in their authored local frames [m]. Only the pad collision
    meshes change; visual geometry, plastic colliders and inertial properties
    remain native. The thumb's open back receives an interior closing surface.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        normals = mesh.GetNormalsAttr().Get()
        if (
            normals is not None
            and mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
            and len(normals) != len(mesh.GetFaceVertexIndicesAttr().Get())
        ):
            # Two native shell meshes retain normals from an older topology.
            # Let the renderer derive normals without changing their geometry.
            mesh.GetNormalsAttr().Block()
    cache = UsdGeom.XformCache()
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        body = root.GetChild(f"right_{finger}_elastomer")
        meshes = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh)]
        visuals = [prim for prim in meshes if not prim.HasAPI(UsdPhysics.CollisionAPI)]
        colliders = [prim for prim in meshes if prim.HasAPI(UsdPhysics.CollisionAPI)]
        if len(visuals) != 1 or len(colliders) != 1:
            raise ValueError(f"Expected one visual and one collision mesh for {body.GetPath()}.")
        visual, collider = UsdGeom.Mesh(visuals[0]), UsdGeom.Mesh(colliders[0])
        surface = create_trimesh_from_geom_mesh(visual.GetPrim())
        if not surface.is_watertight:
            if finger != "thumb":
                raise ValueError(f"Unexpected open Sharpa fingertip: {visual.GetPath()}.")
            surface = _close_thumb_back(surface)
        if not surface.is_volume:
            raise ValueError(f"Sharpa pad has invalid volume or winding: {visual.GetPath()}.")

        # Sibling mesh transforms need not be identical in future asset revisions.
        transform = (
            cache.GetLocalToWorldTransform(visual.GetPrim())
            * cache.GetLocalToWorldTransform(collider.GetPrim()).GetInverse()
        )
        points = np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in surface.vertices], dtype=np.float32)
        collider.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        collider.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(surface.faces), 3, dtype=np.int32)))
        collider.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(surface.faces.reshape(-1).astype(np.int32)))
        collider.GetNormalsAttr().Block()
        # Per-face data from the coarse collider cannot describe the replacement topology.
        primvars = UsdGeom.PrimvarsAPI(collider.GetPrim())
        for name in ("normals", "st"):
            if primvars.HasPrimvar(name):
                primvars.BlockPrimvar(f"primvars:{name}")
        collider.CreateSubdivisionSchemeAttr("none")
        collider.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray((points.min(0), points.max(0)))))
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("none")
    return root


def _close_thumb_back(surface: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close the native thumb pad's single back opening without moving its outer surface."""
    edges = surface.edges
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    vertices, incidence = np.unique(boundary, return_counts=True)
    if not len(vertices) or not np.all(incidence == 2):
        raise ValueError("The Sharpa thumb pad does not have a simple closing contour.")
    following = dict(boundary.tolist())
    visited = set()
    vertex = int(vertices[0])
    while vertex not in visited:
        visited.add(vertex)
        vertex = following[vertex]
    if len(visited) != len(vertices):
        raise ValueError("The Sharpa thumb pad has more than one open contour.")
    # Reversing the existing directed boundary edges preserves winding. The
    # added center lies inside their convex hull, never outside the visual bounds.
    center = len(surface.vertices)
    points = np.vstack((surface.vertices, surface.vertices[vertices].mean(axis=0)))
    cap = np.column_stack((boundary[:, 1], boundary[:, 0], np.full(len(boundary), center)))
    closed = trimesh.Trimesh(vertices=points, faces=np.vstack((surface.faces, cap)), process=False)
    if not closed.is_watertight or not closed.is_winding_consistent:
        raise ValueError("Could not close the Sharpa thumb pad without changing its surface.")
    return closed
