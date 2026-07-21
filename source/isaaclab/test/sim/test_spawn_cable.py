# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg

pytestmark = pytest.mark.unit


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def test_spawn_cable_authors_newton_import_contract(stage):
    points = [(0.0, 0.0, 0.0), (0.2, 0.1, 0.0), (0.4, 0.0, 0.0)]
    material = CableMaterialCfg()
    cfg = CableCfg(
        positions=points,
        physics_material=material,
    )

    root_prim = cfg.func("/World/Cable", cfg)
    curve_prim = stage.GetPrimAtPath("/World/Cable/geometry/mesh")
    curves = UsdGeom.BasisCurves(curve_prim)

    assert root_prim.GetTypeName() == "Xform"
    assert curves
    for point, expected in zip(curves.GetPointsAttr().Get(), points, strict=True):
        assert tuple(point) == pytest.approx(expected)
    assert list(curves.GetCurveVertexCountsAttr().Get()) == [len(points)]
    assert curves.GetTypeAttr().Get() == UsdGeom.Tokens.linear
    assert curves.GetWrapAttr().Get() == UsdGeom.Tokens.nonperiodic
    assert list(curves.GetWidthsAttr().Get()) == pytest.approx([material.thickness])
    assert curves.GetWidthsInterpolation() == UsdGeom.Tokens.constant
    assert "PhysicsCurvesDeformableSimAPI" in curve_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert not curve_prim.HasAPI(UsdPhysics.CollisionAPI)

    physics_binding = UsdShade.MaterialBindingAPI(curve_prim).GetDirectBinding("physics")
    assert physics_binding.GetMaterialPath() == "/World/Cable/geometry/physics_material"
    assert physics_binding.GetMaterialPurpose() == "physics"

    physics_material_prim = stage.GetPrimAtPath(physics_binding.GetMaterialPath())
    assert physics_material_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert "PhysicsCurvesDeformableMaterialAPI" in physics_material_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert physics_material_prim.GetAttribute("physics:thickness").Get() == pytest.approx(0.001)
    assert physics_material_prim.GetAttribute("physics:density").Get() == pytest.approx(1000.0)
    assert physics_material_prim.GetAttribute("physics:stretchStiffness").Get() == pytest.approx(1.0e9)
    assert physics_material_prim.GetAttribute("physics:bendStiffness").Get() == pytest.approx(1.0e6)
