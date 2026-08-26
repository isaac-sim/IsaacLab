# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal stable real-PhysX probes for surface and volume deformables."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import DeformableObject
from isaaclab_physx.sim import (
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

from pxr import Gf, Sdf, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.integration


def _add_api_schemas(prim, schemas: list[str]) -> None:
    schemas_op = Sdf.TokenListOp()
    schemas_op.explicitItems = schemas
    prim.SetMetadata("apiSchemas", schemas_op)


def _bind_material(body_prim, material_cfg) -> None:
    material_prim = material_cfg.func(f"{body_prim.GetPath()}/material", material_cfg)
    UsdShade.MaterialBindingAPI.Apply(body_prim)
    UsdShade.MaterialBindingAPI(body_prim).Bind(
        UsdShade.Material(material_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _spawn_volume_deformable() -> DeformableObject:
    """Author a five-node, two-tetrahedron volume fixture with no optional mesher."""
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Volume")
    tet_mesh = UsdGeom.TetMesh.Define(stage, "/World/Volume/simulation")
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
    tets = [Gf.Vec4i(0, 1, 2, 3), Gf.Vec4i(1, 2, 3, 4)]
    faces = [
        Gf.Vec3i(0, 2, 1),
        Gf.Vec3i(0, 1, 3),
        Gf.Vec3i(0, 3, 2),
        Gf.Vec3i(1, 2, 4),
        Gf.Vec3i(2, 3, 4),
        Gf.Vec3i(3, 1, 4),
    ]
    tet_mesh.CreatePointsAttr(points)
    tet_mesh.CreateTetVertexIndicesAttr(tets)
    tet_mesh.CreateSurfaceFaceVertexIndicesAttr(faces)
    body_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        ["bindPose"]
    )
    body_prim.CreateAttribute("omniphysics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(points)
    body_prim.CreateAttribute("omniphysics:restTetVtxIndices", Sdf.ValueTypeNames.Int4Array).Set(tets)
    body_prim.CreateAttribute("velocities", Sdf.ValueTypeNames.Vector3fArray).Set([Gf.Vec3f()] * len(points))
    visual = UsdGeom.Mesh.Define(stage, "/World/Volume/visual")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr([3] * len(faces))
    visual.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    _bind_material(body_prim, PhysxDeformableBodyMaterialCfg())
    return DeformableObject(DeformableObjectCfg(prim_path="/World/Volume"))


def _spawn_surface_deformable() -> DeformableObject:
    """Author a four-node, two-triangle surface fixture."""
    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Surface")
    body_prim = mesh.GetPrim()
    _add_api_schemas(
        body_prim,
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsSurfaceDeformableSimAPI",
            "OmniPhysicsDeformablePoseAPI:default",
            "PhysicsCollisionAPI",
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
    _bind_material(
        body_prim,
        PhysxSurfaceDeformableBodyMaterialCfg(
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
        ),
    )
    return DeformableObject(DeformableObjectCfg(prim_path="/World/Surface"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_volume_deformable_real_physx_state_and_kinematic_target() -> None:
    """Prove volume view/material creation plus nodal and kinematic-target writes."""
    with build_simulation_context(device="cuda:0", gravity_enabled=False) as sim:
        deformable = _spawn_volume_deformable()
        sim.reset()

        assert deformable.is_initialized
        assert deformable._deformable_type == "volume"
        assert deformable.material_physx_view is not None
        assert deformable.data.nodal_state_w.torch.shape[-1] == 6
        positions = deformable.data.nodal_pos_w.torch.clone()
        velocities = torch.zeros_like(positions)
        velocities[:, 0, 0] = 0.25
        state = torch.cat((positions, velocities), dim=-1)
        deformable.write_nodal_state_to_sim_index(state)
        torch.testing.assert_close(deformable.data.nodal_state_w.torch, state)

        targets = deformable.data.nodal_kinematic_target.torch.clone()
        targets[:, 0, :3] = positions[:, 0] + torch.tensor([0.01, 0.02, 0.03], device="cuda:0")
        targets[:, 0, 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(targets)
        raw_targets = wp.to_torch(deformable.root_view.get_simulation_nodal_kinematic_targets()).reshape_as(targets)
        torch.testing.assert_close(raw_targets, targets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_surface_deformable_real_physx_state_probe() -> None:
    """Prove surface view/material creation and a nodal-position TensorAPI write."""
    with build_simulation_context(device="cuda:0", gravity_enabled=False) as sim:
        deformable = _spawn_surface_deformable()
        sim.reset()

        assert deformable.is_initialized
        assert deformable._deformable_type == "surface"
        assert deformable.root_view.count == 1
        assert deformable.max_sim_vertices_per_body == 4
        assert deformable.material_physx_view is not None
        positions = deformable.data.nodal_pos_w.torch.clone()
        positions[:, 0, 2] += 0.01
        deformable.write_nodal_pos_to_sim_index(positions)
        raw_positions = wp.to_torch(deformable.root_view.get_simulation_nodal_positions()).reshape_as(positions)
        torch.testing.assert_close(raw_positions, positions)
