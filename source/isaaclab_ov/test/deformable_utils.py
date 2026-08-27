# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test utilities for pre-authored OVPhysX deformables."""

from __future__ import annotations

from isaaclab_physx.sim.spawners.materials import (
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

_VOLUME_MATERIAL_CFG = PhysxDeformableBodyMaterialCfg(
    dynamic_friction=0.5,
    youngs_modulus=1000.0,
    poissons_ratio=0.3,
)

_SURFACE_MATERIAL_CFG = PhysxSurfaceDeformableBodyMaterialCfg(
    density=900.0,
    static_friction=0.35,
    dynamic_friction=0.4,
    youngs_modulus=2000.0,
    poissons_ratio=0.25,
    surface_thickness=0.02,
    surface_stretch_stiffness=0.8,
    surface_shear_stiffness=0.7,
    surface_bend_stiffness=0.6,
    elasticity_damping=0.03,
    bend_damping=0.04,
)


@configclass
class _PreauthoredDeformableSpawnerCfg(sim_utils.SpawnerCfg):
    """Configuration for a pre-authored deformable test fixture."""

    material_path: str | None = "material"
    """Physics material path, relative to the body prim or absolute, or None to omit it."""


def _add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    """Author applied schemas without requiring the Omni PhysX Kit modules."""
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _apply_transform(
    prim: Usd.Prim,
    translation: tuple[float, float, float] | None,
    orientation: tuple[float, float, float, float] | None,
) -> None:
    """Apply an optional rigid transform to an authored deformable prim."""
    xform = UsdGeom.Xformable(prim)
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(orientation[3], orientation[0], orientation[1], orientation[2]))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _bind_material(prim: Usd.Prim, material_cfg: sim_utils.PhysicsMaterialCfg, material_path: str | None) -> None:
    """Spawn and optionally bind a deformable material to an authored body prim."""
    if material_path is None:
        return
    if not material_path.startswith("/"):
        material_path = f"{prim.GetPath()}/{material_path}"
    material_prim = material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(material_prim)
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


@sim_utils.clone
def spawn_pre_tetrahedralized_deformable(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a five-node volume deformable with two authored tetrahedra."""
    stage = sim_utils.get_current_stage()
    root_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    tet_mesh = UsdGeom.TetMesh.Define(stage, f"{prim_path}/simulation")
    body_prim = tet_mesh.GetPrim()
    _add_api_schemas(
        body_prim,
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsVolumeDeformableSimAPI",
            "OmniPhysicsDeformablePoseAPI:default",
            "PhysicsCollisionAPI",
        ],
    )

    points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.2, 0.0, 0.0),
        Gf.Vec3f(0.0, 0.2, 0.0),
        Gf.Vec3f(0.0, 0.0, 0.2),
        Gf.Vec3f(0.2, 0.2, 0.2),
    ]
    tet_indices = [Gf.Vec4i(0, 1, 2, 3), Gf.Vec4i(1, 2, 3, 4)]
    surface_indices = [
        Gf.Vec3i(0, 2, 1),
        Gf.Vec3i(0, 1, 3),
        Gf.Vec3i(0, 3, 2),
        Gf.Vec3i(1, 2, 4),
        Gf.Vec3i(2, 3, 4),
        Gf.Vec3i(3, 1, 4),
    ]
    body_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    visual_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/visual")
    visual_mesh.CreatePointsAttr(points)
    visual_mesh.CreateFaceVertexCountsAttr([3] * len(surface_indices))
    visual_mesh.CreateFaceVertexIndicesAttr([vertex_index for face in surface_indices for vertex_index in face])
    body_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        ["bindPose"]
    )
    body_prim.CreateAttribute("omniphysics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("omniphysics:restTetVtxIndices", Sdf.ValueTypeNames.Int4Array).Set(tet_indices)
    body_prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("surfaceFaceVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(surface_indices)
    body_prim.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(tet_indices)
    body_prim.CreateAttribute("velocities", Sdf.ValueTypeNames.Vector3fArray).Set([Gf.Vec3f()] * len(points))

    _apply_transform(root_prim, translation, orientation)
    _bind_material(body_prim, _VOLUME_MATERIAL_CFG, cfg.material_path)
    return root_prim


@sim_utils.clone
def spawn_pretriangulated_surface_deformable(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a four-node surface deformable with two authored triangles."""
    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    body_prim = mesh.GetPrim()
    _add_api_schemas(
        body_prim,
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsSurfaceDeformableSimAPI",
            "OmniPhysicsDeformablePoseAPI:default",
            "PhysicsCollisionAPI",
            "MaterialBindingAPI",
        ],
    )

    points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.2, 0.0, 0.0),
        Gf.Vec3f(0.2, 0.2, 0.0),
        Gf.Vec3f(0.0, 0.2, 0.0),
    ]
    triangles = [Gf.Vec3i(0, 1, 2), Gf.Vec3i(0, 2, 3)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    body_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        ["bindPose"]
    )
    body_prim.CreateAttribute("omniphysics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("omniphysics:restTriVtxIndices", Sdf.ValueTypeNames.Int3Array).Set(triangles)
    body_prim.CreateAttribute("velocities", Sdf.ValueTypeNames.Vector3fArray).Set([Gf.Vec3f()] * len(points))

    _apply_transform(body_prim, translation, orientation)
    _bind_material(body_prim, _SURFACE_MATERIAL_CFG, cfg.material_path)
    return body_prim


def pre_tetrahedralized_deformable_spawn_cfg(
    material_path: str | None = "material",
) -> sim_utils.SpawnerCfg:
    """Create the pre-tetrahedralized volume-deformable test spawner."""
    return _PreauthoredDeformableSpawnerCfg(func=spawn_pre_tetrahedralized_deformable, material_path=material_path)


def pretriangulated_surface_deformable_spawn_cfg() -> sim_utils.SpawnerCfg:
    """Create the pretriangulated surface-deformable test spawner."""
    return _PreauthoredDeformableSpawnerCfg(func=spawn_pretriangulated_surface_deformable)
