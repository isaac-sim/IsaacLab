# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD spawning and collider adaptations for the assemble trocar task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import clone, get_current_stage

if TYPE_CHECKING:
    from pxr import Sdf, Usd

    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


@clone
def spawn_usd_with_newton_compatible_colliders(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD asset and adapt its authored PhysX colliders only when running on Newton.

    The healthcare G1 and surgical tray assets use two USD patterns that PhysX supports but Newton's USD importer
    does not: mesh-merge collisions authored on Xforms and mirrored collider meshes with negative scales. Applying
    the equivalent geometry edits here confines the compatibility conversion to this task and leaves PhysX unchanged.
    """
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file

    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    sim = SimulationContext.instance()
    physics_modules = (
        {cls.__module__.split(".")[0] for cls in type(sim.cfg.physics).__mro__} if sim is not None else set()
    )
    if "isaaclab_newton" in physics_modules:
        stage = get_current_stage()
        _expand_merged_mesh_colliders(stage, prim.GetPath())
        _bake_negative_mesh_collider_scales(stage, prim.GetPath())
    return prim


def _expand_merged_mesh_colliders(stage: Usd.Stage, root_path: Sdf.Path) -> int:
    """Move Xform collision APIs onto descendant meshes for Newton's USD importer."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    targets = [
        prim.GetPath()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and not prim.IsA(UsdGeom.Gprim)
        and any(child.IsA(UsdGeom.Mesh) for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    ]

    num_meshes = 0
    for path in targets:
        _make_ancestors_editable(stage, path)
        prim = stage.GetPrimAtPath(path)
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        approximation = (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )
        for mesh in Usd.PrimRange(prim):
            if not mesh.IsA(UsdGeom.Mesh) or mesh.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(enabled)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
            if approximation is not None:
                mesh_collision.CreateApproximationAttr(approximation)
            num_meshes += 1
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    return num_meshes


def _bake_negative_mesh_collider_scales(stage: Usd.Stage, root_path: Sdf.Path) -> int:
    """Move collider scale signs into mesh points so MJWarp receives positive geom sizes."""
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    root = stage.GetPrimAtPath(root_path)
    targets = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if ops and ops[-1].GetOpType() == UsdGeom.XformOp.TypeScale:
            scale = ops[-1].Get()
            if (
                scale is not None
                and UsdGeom.Mesh(prim).GetPointsAttr().Get() is not None
                and any(s < 0.0 for s in scale)
            ):
                targets.append(prim.GetPath())

    for path in targets:
        _make_ancestors_editable(stage, path)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        scale_op = UsdGeom.Xformable(mesh).GetOrderedXformOps()[-1]
        scale = np.asarray(scale_op.Get(), dtype=np.float64)
        sign = np.sign(scale)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) * sign
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        if np.prod(sign) < 0.0:
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
            flipped = np.concatenate([indices[offset : offset + count][::-1] for offset, count in zip(offsets, counts)])
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(flipped.astype(np.int32)))
        scale_op.Set(type(scale_op.Get())(*np.abs(scale)))
    return len(targets)


def _make_ancestors_editable(stage: Usd.Stage, path: Sdf.Path) -> None:
    """Disable instancing on a target's ancestors before authoring below it."""
    for prefix in path.GetPrefixes():
        ancestor = stage.GetPrimAtPath(prefix)
        if ancestor.IsInstance():
            ancestor.SetInstanceable(False)
