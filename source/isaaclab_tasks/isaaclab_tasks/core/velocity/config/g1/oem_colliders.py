# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose Lab G1 collision profiles over Unitree's floating Dex3 USD."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_assets.robots.unitree_g1_oem import retrieve_g1_oem_usd

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim import UsdFileCfg


LAB_G1_29DOF_USD = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd"

_BODY_SHAPES = (
    ("torso_link", "collisions/torso_link_rev_1_0/mesh", "lab_torso_collision"),
    ("torso_link", "collisions/head_link/mesh", "lab_head_collision"),
    ("pelvis", "collisions/pelvis_contour_link/mesh", "lab_pelvis_collision"),
)


def _copy_convex_hull(stage: Usd.Stage, source: Usd.Prim, destination, transform):
    import numpy as np
    from scipy.spatial import ConvexHull

    from pxr import UsdGeom, UsdPhysics, Vt

    points = np.asarray(UsdGeom.Mesh(source).GetPointsAttr().Get())
    hull = ConvexHull(points)
    triangles = hull.simplices.copy()
    faces = points[triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    flip = np.einsum("ij,ij->i", hull.equations[:, :3], normals) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    indices = np.empty(len(points), dtype=np.int32)
    indices[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)

    mesh = UsdGeom.Mesh.Define(stage, destination)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points[hull.vertices], dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(indices[triangles].reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.AddTransformOp().Set(transform)
    shape = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(shape).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(shape).CreateApproximationAttr("convexHull")


def _enabled_colliders(stage: Usd.Stage) -> list[Usd.Prim]:
    from pxr import Usd, UsdPhysics

    return [
        prim
        for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    ]


def _author_minimal_colliders(stage: Usd.Stage, lab: Usd.Stage) -> int:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim()
    lab_root = lab.GetDefaultPrim()
    transforms = UsdGeom.XformCache()
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        lab_body = lab_root.GetChild(body_name)
        lab_plate = UsdGeom.Cube(lab_body.GetChild("foot_plate"))
        transform, _ = transforms.ComputeRelativeTransform(lab_plate.GetPrim(), lab_body)
        plate = UsdGeom.Cube.Define(stage, root.GetChild(body_name).GetPath().AppendChild("foot_plate"))
        plate.CreateSizeAttr(lab_plate.GetSizeAttr().Get())
        plate.AddTransformOp().Set(transform)
        UsdPhysics.CollisionAPI.Apply(plate.GetPrim()).CreateCollisionEnabledAttr(True)

    for body_name, source_name, destination_name in _BODY_SHAPES:
        lab_body = lab_root.GetChild(body_name)
        mesh = lab.GetPrimAtPath(lab_body.GetPath().AppendPath(source_name))
        transform, _ = transforms.ComputeRelativeTransform(mesh, lab_body)
        destination = root.GetChild(body_name).GetPath().AppendChild(destination_name)
        _copy_convex_hull(stage, mesh, destination, transform)
    return 5


def _author_full_colliders(stage: Usd.Stage, lab: Usd.Stage) -> int:
    from pxr import Usd, UsdGeom, UsdPhysics

    bodies = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    transforms = UsdGeom.XformCache()
    sources = _enabled_colliders(lab)
    for index, source in enumerate(sources):
        body = source.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or body.GetName() not in bodies:
            raise ValueError(f"Could not match a G1 body for collider {source.GetPath()}.")

        name = source.GetName()
        if name != "foot_plate":
            name = f"lab_full_{index}_{name}"
        destination = bodies[body.GetName()].GetPath().AppendChild(name)
        target = stage.DefinePrim(destination, source.GetTypeName())
        for attr_name in ("points", "faceVertexCounts", "faceVertexIndices", "subdivisionScheme", "size"):
            attr = source.GetAttribute(attr_name)
            if attr and (value := attr.Get()) is not None:
                target.GetAttribute(attr_name).Set(value)
        UsdPhysics.CollisionAPI.Apply(target).CreateCollisionEnabledAttr(True)
        if source.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(source).GetApproximationAttr().Get()
            UsdPhysics.MeshCollisionAPI.Apply(target).CreateApproximationAttr(approximation)
        transform, _ = transforms.ComputeRelativeTransform(source, body)
        UsdGeom.Xformable(target).AddTransformOp().Set(transform)
        UsdGeom.Imageable(target).MakeInvisible()

    return len(sources)


@cache
def _build_oem_collision_layer(
    oem_uri: str, profile: Literal["minimal", "full"], *, match_depth_action_scale: bool = False
) -> TemporaryDirectory:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    lab = Usd.Stage.Open(retrieve_file_path(LAB_G1_29DOF_USD))
    with Usd.EditContext(lab, lab.GetSessionLayer()):
        lab.GetDefaultPrim().GetVariantSet("Colliders").SetVariantSelection(profile)
    if lab.GetCompositionErrors():
        raise ValueError(f"Could not compose the G1 collision source: {lab.GetCompositionErrors()}")

    source = retrieve_g1_oem_usd(oem_uri)
    oem = Usd.Stage.Open(source)
    layer = Sdf.Layer.CreateAnonymous("g1_oem_colliders.usda")
    layer.subLayerPaths = [source]
    layer.defaultPrim = oem.GetDefaultPrim().GetName()
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(oem))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(oem))
    for prim in _enabled_colliders(stage):
        stage.OverridePrim(prim.GetPath()).SetActive(False)

    author = _author_minimal_colliders if profile == "minimal" else _author_full_colliders
    expected_count = author(stage, lab)
    if match_depth_action_scale:
        # Depth distillation uses the Lab teacher's ankle and waist action scales.
        lab_joints = lab.GetDefaultPrim().GetChild("joints")
        oem_joints = stage.GetDefaultPrim().GetChild("joints")
        for name in (
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
        ):
            max_force = UsdPhysics.DriveAPI.Get(lab_joints.GetChild(name), "angular").GetMaxForceAttr().Get()
            UsdPhysics.DriveAPI.Get(oem_joints.GetChild(name), "angular").GetMaxForceAttr().Set(max_force)

    if stage.GetCompositionErrors():
        raise ValueError(f"Could not compose the OEM G1 collision layer: {stage.GetCompositionErrors()}")
    if len(_enabled_colliders(stage)) != expected_count:
        raise ValueError(f"Expected {expected_count} collision shapes in the OEM G1 {profile} profile.")

    temporary = TemporaryDirectory(prefix="isaaclab-g1-oem-")
    if not layer.Export(str(Path(temporary.name) / "g1_oem_colliders.usdc")):
        raise RuntimeError("Could not export the OEM G1 collision layer.")
    return temporary


def get_oem_minimal_usd(oem_uri: str, *, match_depth_action_scale: bool = False) -> str:
    """Return the composed asset used by the locomotion spawner and task startup."""
    layer = _build_oem_collision_layer(oem_uri, "minimal", match_depth_action_scale=match_depth_action_scale)
    return str(Path(layer.name) / "g1_oem_colliders.usdc")


def spawn_g1_oem_minimal(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Spawn the OEM asset with the five contact shapes used by rough locomotion."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    local_cfg = replace(cfg, usd_path=get_oem_minimal_usd(cfg.usd_path))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)


def spawn_g1_oem_depth_student(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Spawn the OEM asset with the minimal colliders and Lab teacher's action-scale references."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    local_cfg = replace(cfg, usd_path=get_oem_minimal_usd(cfg.usd_path, match_depth_action_scale=True))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)


def spawn_g1_oem_full(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Spawn the OEM asset with the full contact profile used by the terrain courses."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    asset_dir = _build_oem_collision_layer(cfg.usd_path, "full").name
    local_cfg = replace(cfg, usd_path=str(Path(asset_dir) / "g1_oem_colliders.usdc"))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)
