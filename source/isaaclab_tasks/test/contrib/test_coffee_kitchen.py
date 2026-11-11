# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hosted kitchen preparation at the shared USD spawning boundary."""

from pathlib import Path

import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaaclab.sim import use_stage

from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import spawn_kitchen
from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env_cfg import CoffeeKitchenCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


@pytest.fixture
def kitchen_file(tmp_path: Path) -> str:
    """A kitchen containing the import defects and authored presentation policies."""
    path = tmp_path / "kitchen.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Kitchen")
    root.AddTranslateOp().Set(Gf.Vec3d(0.5, -0.2, 0.1))
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Kitchen/Appliance/Mesh")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 3)
    mesh.GetPrim().CreateAttribute("primvars:normals", Sdf.ValueTypeNames.TokenArray).Set(["vertex"])
    door = UsdGeom.Xform.Define(stage, "/Kitchen/Appliance/Door").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(door).CreateRigidBodyEnabledAttr(False)
    UsdPhysics.FixedJoint.Define(stage, "/Kitchen/Appliance/Joint")
    UsdGeom.Camera.Define(stage, "/Kitchen/Camera")
    UsdLux.DomeLight.Define(stage, "/Kitchen/Light").CreateIntensityAttr(25.0)
    UsdGeom.Xform.Define(stage, "/Kitchen/KitchenRoom/Ceiling")
    UsdGeom.BasisCurves.Define(stage, "/Kitchen/Appliance/grasp_identifier_curve")
    UsdGeom.BasisCurves.Define(stage, "/Kitchen/DecorativeCurve")
    stage.GetRootLayer().Save()
    return str(path)


@pytest.mark.parametrize(("disable_lights", "remove_ceiling"), [(False, False), (True, False), (False, True)])
def test_kitchen_prepares_every_clone_without_changing_other_environments(
    kitchen_file: str, disable_lights: bool, remove_ceiling: bool
) -> None:
    """Repair native import while keeping lighting/ceiling choices independent and floors local."""
    stage = Usd.Stage.CreateInMemory()
    for index in range(3):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}")
        UsdGeom.Mesh.Define(stage, f"/World/envs/env_{index}/Table/StudioFloor")
    cfg = CoffeeKitchenCfg(usd_path=kitchen_file, disable_lights=disable_lights, remove_ceiling=remove_ceiling)
    with use_stage(stage):
        spawn_kitchen("/World/envs/env_[01]/Kitchen", cfg, translation=(0.0, 0.0, 0.0))
    for index in range(2):
        path = f"/World/envs/env_{index}"
        root = stage.GetPrimAtPath(path + "/Kitchen")
        translation = UsdGeom.Xformable(root).GetLocalTransformation().ExtractTranslation()
        assert tuple(translation) == pytest.approx((0.5, -0.2, 0.1))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path + "/Kitchen/Appliance/Mesh"))
        assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
        assert mesh.GetNormalsAttr().Get() == [(0.0, 0.0, 1.0)] * 3
        assert not mesh.GetPrim().GetAttribute("primvars:normals").Get()
        assert not stage.GetPrimAtPath(path + "/Kitchen/Appliance/Door").HasAPI(UsdPhysics.RigidBodyAPI)
        assert not stage.GetPrimAtPath(path + "/Kitchen/Appliance/Joint").IsActive()
        assert not stage.GetPrimAtPath(path + "/Kitchen/Camera").IsActive()
        assert not stage.GetPrimAtPath(path + "/Kitchen/Appliance/grasp_identifier_curve").IsActive()
        assert stage.GetPrimAtPath(path + "/Kitchen/DecorativeCurve").IsActive()
        light = stage.GetPrimAtPath(path + "/Kitchen/Light")
        assert light.IsActive() is not disable_lights
        assert light.GetAttribute("inputs:intensity").Get() == 25.0
        assert stage.GetPrimAtPath(path + "/Kitchen/KitchenRoom/Ceiling").IsActive() is not remove_ceiling
        assert (
            UsdGeom.Imageable(stage.GetPrimAtPath(path + "/Table/StudioFloor")).GetVisibilityAttr().Get() == "invisible"
        )
    assert not stage.GetPrimAtPath("/World/envs/env_2/Kitchen").IsValid()
    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_2/Table/StudioFloor")).GetVisibilityAttr().Get()
        != "invisible"
    )
