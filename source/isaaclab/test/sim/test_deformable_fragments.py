# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deformable creation, tuning, and spawner targeting."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg, UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.materials import OmniPhysicsSurfaceDeformableMaterialCfg

pytestmark = pytest.mark.integration


@pytest.fixture
def stage():
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(device="cpu"))
    yield sim_utils.get_current_stage()
    SimulationContext.clear_instance()


def _volume_asset(stage, path):
    body = UsdGeom.Xform.Define(stage, path).GetPrim()
    mesh = UsdGeom.TetMesh.Define(stage, path + "/tet")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    mesh.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    return body


def test_create_and_modify_deformable(stage, caplog):
    """Reuse a supplied TetMesh, tune without rebuilding, and report missing targets."""
    stage = Usd.Stage.CreateInMemory()
    body = _volume_asset(stage, "/World/Soft")
    assert sim_utils.apply_volume_deformable_properties(
        "/World/Soft", [OmniPhysicsDeformableBodyCfg(mass=0.5)], create_if_missing=True, stage=stage
    )
    assert sim_utils.has_deformable_body_api(body)
    mesh = stage.GetPrimAtPath("/World/Soft/tet")
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert body.GetAttribute("omniphysics:mass").Get() == pytest.approx(0.5)
    points = mesh.GetAttribute("points").Get()
    assert sim_utils.apply_volume_deformable_properties(
        "/World/Soft", [OmniPhysicsDeformableBodyCfg(mass=1.0)], stage=stage
    )
    assert mesh.GetAttribute("points").Get() == points
    assert not stage.GetPrimAtPath("/World/Soft/sim_mesh")
    assert body.GetAttribute("omniphysics:mass").Get() == pytest.approx(1.0)
    assert not sim_utils.apply_volume_deformable_properties(
        "/World/Missing", [OmniPhysicsDeformableBodyCfg()], stage=stage
    )
    assert "No deformable-body targets matched" in caplog.text


def test_deformable_type_respects_nested_bodies(stage, caplog):
    """Reject the wrong family without mistaking a nested body's mesh for its parent's."""
    outer = UsdGeom.Xform.Define(stage, "/World/Outer").GetPrim()
    nested = UsdGeom.Xform.Define(stage, "/World/Outer/A_nested").GetPrim()
    for body, kind in ((nested, "Surface"), (outer, "Volume")):
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        UsdGeom.Mesh.Define(stage, str(body.GetPath()) + "/sim_mesh").GetPrim().AddAppliedSchema(
            f"Physics{kind}DeformableSimAPI"
        )
    assert not sim_utils.apply_volume_deformable_properties(
        "/World/Outer(/A_nested)?", [OmniPhysicsDeformableBodyCfg(mass=1.0)], create_if_missing=True, stage=stage
    )
    assert outer.GetAttribute("omniphysics:mass").Get() == pytest.approx(1.0)
    assert not nested.GetAttribute("omniphysics:mass").IsValid()
    assert "surface" in caplog.text and "volume" in caplog.text
    assert sim_utils.apply_surface_deformable_properties(
        "/World/Outer/A_nested", [OmniPhysicsDeformableBodyCfg(mass=2.0)], stage=stage
    )
    assert nested.GetAttribute("omniphysics:mass").Get() == pytest.approx(2.0)


@pytest.mark.parametrize("props", [OmniPhysicsDeformableBodyCfg(mass=0.2), [], {}])
def test_mesh_surface_deformable_spawn(stage, props):
    """Bare and empty slots create one body; collision fragments target its simulation mesh."""
    cfg = sim_utils.MeshRectangleCfg(
        size=(0.1, 0.1),
        surface_deformable_props=props,
        physics_material=OmniPhysicsSurfaceDeformableMaterialCfg(surface_thickness=0.01),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=False),
    )
    cfg.func("/World/Cloth", cfg)
    body = stage.GetPrimAtPath("/World/Cloth")
    assert sim_utils.has_deformable_body_api(body)
    mesh = stage.GetPrimAtPath("/World/Cloth/sim_mesh")
    assert mesh.GetAttribute("physics:collisionEnabled").Get() is False
    assert not body.HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath("/World/Cloth/geometry/mesh/sim_mesh")
    if isinstance(props, OmniPhysicsDeformableBodyCfg):
        assert body.GetAttribute("omniphysics:mass").Get() == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"volume_deformable_props": []}, "one deformable"),
        ({"deformable_props": sim_utils.DeformableBodyPropertiesBaseCfg}, "one deformable"),
        ({"rigid_props": sim_utils.UsdPhysicsRigidBodyCfg()}, "both deformable and rigid"),
        ({"collision_props": sim_utils.CollisionBaseCfg()}, "collision fragments"),
        ({"mesh_collision_props": sim_utils.UsdPhysicsMeshCollisionCfg()}, "mesh_collision_props"),
    ],
)
@pytest.mark.filterwarnings("ignore:DeformableBodyPropertiesBaseCfg is deprecated:DeprecationWarning")
def test_mesh_rejects_conflicting_deformable_properties(stage, kwargs, message):
    if "deformable_props" in kwargs:
        kwargs = {"deformable_props": kwargs["deformable_props"]()}
    cfg = sim_utils.MeshRectangleCfg(size=(0.1, 0.1), surface_deformable_props=[], **kwargs)
    with pytest.raises(ValueError, match=message):
        cfg.func("/World/Bad", cfg)


def test_usd_file_deformable_targets_only_spawn_prim(stage, tmp_path):
    asset = Usd.Stage.CreateNew(str(tmp_path / "tet.usda"))
    asset.SetDefaultPrim(_volume_asset(asset, "/Asset"))
    asset.Save()
    cfg = sim_utils.UsdFileCfg(
        usd_path=asset.GetRootLayer().identifier, volume_deformable_props=OmniPhysicsDeformableBodyCfg(mass=0.5)
    )
    cfg.func("/World/Soft", cfg)
    assert sim_utils.has_deformable_body_api(stage.GetPrimAtPath("/World/Soft"))
    assert stage.GetPrimAtPath("/World/Soft/tet").HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath("/World/Soft/sim_mesh")
    assert not sim_utils.has_deformable_body_api(stage.GetPrimAtPath("/World/Soft/tet"))
    cfg = cfg.replace(mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg())
    with pytest.raises(ValueError, match="mesh_collision_props"):
        cfg.func("/World/Rejected", cfg)
    assert not stage.GetPrimAtPath("/World/Rejected")
