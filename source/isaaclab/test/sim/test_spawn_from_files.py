# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawning from files needs a USD stage but no physics. Only the URDF importer may need Kit."""

import os

from isaaclab.app import AppLauncher
from isaaclab.utils.version import standalone_importers_available

# Prefer kit-less; fall back to Kit when the standalone importers are not usable.
_USE_KIT = not standalone_importers_available() and AppLauncher.is_available()
simulation_app = AppLauncher(headless=True).app if _USE_KIT else None

"""Rest everything follows."""

import pytest
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]

GELSIGHT_USD = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def test_spawn_usd(stage):
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd")
    prim = cfg.func("/World/Franka", cfg)
    assert prim.GetPath() == "/World/Franka"
    assert prim.GetTypeName() == "Xform"

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda2_instanceable.usd")
    with pytest.raises(FileNotFoundError):
        cfg.func("/World/Missing", cfg)


def test_spawn_usd_make_uninstanceable_applies_material_to_instance_colliders(stage, tmp_path):
    geometry_path = tmp_path / "geometry.usda"
    geometry_stage = Usd.Stage.CreateNew(str(geometry_path))
    geometry_stage.SetDefaultPrim(UsdGeom.Xform.Define(geometry_stage, "/Geometry").GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(geometry_stage, "/Geometry/Collider").GetPrim())
    geometry_stage.GetRootLayer().Save()

    asset_path = tmp_path / "asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    asset_stage.SetDefaultPrim(UsdGeom.Xform.Define(asset_stage, "/Asset").GetPrim())
    instance = UsdGeom.Xform.Define(asset_stage, "/Asset/collisions").GetPrim()
    instance.GetReferences().AddReference(str(geometry_path), "/Geometry")
    instance.SetInstanceable(True)
    asset_stage.GetRootLayer().Save()

    cfg = sim_utils.UsdFileCfg(
        usd_path=str(asset_path),
        make_uninstanceable=True,
        physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.73),
    )
    cfg.func("/World/Asset", cfg)

    collider = stage.GetPrimAtPath("/World/Asset/collisions/Collider")
    assert not stage.GetPrimAtPath("/World/Asset/collisions").IsInstance()
    assert not collider.IsInstanceProxy()
    material = stage.GetPrimAtPath("/World/Asset/material")
    bound_material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
    assert bound_material.GetPath() == material.GetPath()
    assert material.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.73)


def test_spawn_urdf(stage):
    if _USE_KIT:
        sim_utils.enable_extension("isaacsim.asset.importer.urdf")
        extension_path = sim_utils.get_extension_path("isaacsim.asset.importer.urdf")
        asset_path = f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
    else:
        asset_path = os.path.join(
            os.path.dirname(isaaclab.__file__), "controllers", "config", "data", "lula_franka_gen.urdf"
        )
    cfg = sim_utils.UrdfFileCfg(
        asset_path=asset_path,
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    prim = cfg.func("/World/Franka", cfg)
    assert prim.GetPath() == "/World/Franka"
    assert prim.GetTypeName() == "Xform"


def test_spawn_ground_plane(stage):
    cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(10.0, 20.0))
    prim = cfg.func("/World/ground_plane", cfg)
    assert prim.GetPath() == "/World/ground_plane"
    assert prim.GetTypeName() == "Xform"

    # the UVs are rescaled with the plane so the tiles stay metric
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ground_plane/Environment/Geometry"))
    assert [tuple(uv) for uv in UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()] == [
        (-2.5, -5.0),
        (2.5, -5.0),
        (2.5, 5.0),
        (-2.5, 5.0),
    ]
    shader = UsdShade.Shader.Get(stage, "/World/ground_plane/Looks/theGrid/Shader")
    assert tuple(shader.GetInput("diffuse_tint").Get()) == pytest.approx((0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="already exists"):
        cfg.func("/World/ground_plane", cfg)


@pytest.mark.parametrize(
    "physics_material_prim_path",
    ["elastomer", ["elastomer", "gelsight_finger"], None],
    ids=["single", "multiple", "none"],
)
def test_spawn_usd_with_compliant_contact_material(stage, physics_material_prim_path):
    cfg = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=GELSIGHT_USD,
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True),
        compliant_contact_stiffness=1000.0,
        compliant_contact_damping=100.0,
        physics_material_prim_path=physics_material_prim_path,
    )
    prim = cfg.func("/World/Robot", cfg)
    assert prim.GetPath() == "/World/Robot"

    expected_links = physics_material_prim_path or []
    if isinstance(expected_links, str):
        expected_links = [expected_links]
    for link_name in ("elastomer", "gelsight_finger"):
        material_prim = stage.GetPrimAtPath(f"/World/Robot/{link_name}/compliant_material")
        assert material_prim.IsValid() == (link_name in expected_links)
        if material_prim.IsValid():
            assert material_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == 1000.0
            assert material_prim.GetAttribute("physxMaterial:compliantContactDamping").Get() == 100.0
