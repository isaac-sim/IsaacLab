# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless coverage of moving an asset's nested rigid body onto the spawned prim."""

import pytest

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils


def _write_asset(
    path, *, body_offset=(0.0, 0.0, 0.0), center_of_mass=None, second_body=False, instanced_body=False
) -> str:
    """Write an asset whose rigid body sits at ``/Root/Geometry/body`` above a collider with its own mass.

    ``/Root/Geometry`` references its content from ``/Prototype`` so that it can be made an instance.
    """
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    stage.CreateClassPrim("/Prototype")
    body = UsdGeom.Xform.Define(stage, "/Prototype/body")
    if body_offset != (0.0, 0.0, 0.0):
        body.AddTranslateOp().Set(Gf.Vec3d(*body_offset))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(True)
    body_mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    body_mass.CreateMassAttr(0.3)
    if center_of_mass is not None:
        body_mass.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    collider = UsdGeom.Cube.Define(stage, "/Prototype/body/collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.MassAPI.Apply(collider).CreateMassAttr(0.1)
    if second_body:
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Prototype/lid").GetPrim())
    geometry = UsdGeom.Scope.Define(stage, "/Root/Geometry").GetPrim()
    geometry.GetReferences().AddInternalReference("/Prototype")
    geometry.SetInstanceable(instanced_body)
    stage.Save()
    return str(path)


@pytest.fixture
def stage():
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        yield stage


def _reference(stage, asset_path: str) -> Usd.Prim:
    prim = stage.DefinePrim("/World/Object", "Xform")
    prim.GetReferences().AddReference(asset_path)
    return prim


def test_moves_body_schemas_and_values_onto_prim(stage, tmp_path):
    """The body and its own mass move onto the prim; the collider keeps its collider mass."""
    prim = _reference(stage, _write_asset(tmp_path / "asset.usda"))

    sim_utils.move_rigid_body_to_root("/World/Object", stage=stage)

    body = stage.GetPrimAtPath("/World/Object/Geometry/body")
    collider = stage.GetPrimAtPath("/World/Object/Geometry/body/collider")
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI) and not body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True
    assert prim.HasAPI(UsdPhysics.MassAPI) and not body.HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(0.3)
    assert collider.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MassAPI(collider).GetMassAttr().Get() == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("asset_kwargs", "match"),
    [
        ({"second_body": True}, "Expected one rigid body"),
        ({"instanced_body": True}, "inside an instance"),
    ],
    ids=["two-bodies", "instanced-body"],
)
def test_refuses_moves_that_change_physics(stage, tmp_path, asset_kwargs, match):
    """A move that would merge bodies or edit an instance proxy is refused."""
    _reference(stage, _write_asset(tmp_path / "asset.usda", **asset_kwargs))

    with pytest.raises(ValueError, match=match):
        sim_utils.move_rigid_body_to_root("/World/Object", stage=stage)


def test_moves_offset_body_and_keeps_its_center_of_mass(stage, tmp_path):
    """An offset body moves onto the prim, and its center of mass keeps its place in the prim frame."""
    asset = _write_asset(tmp_path / "asset.usda", body_offset=(0.0, 0.0, 0.1), center_of_mass=(0.01, 0.0, 0.0))
    prim = _reference(stage, asset)
    sim_utils.move_rigid_body_to_root("/World/Object", stage=stage)
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get() == pytest.approx((0.01, 0.0, 0.1))
