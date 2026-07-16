# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg


@pytest.fixture
def stage():
    """Create an in-memory USD stage."""
    return sim_utils.create_new_stage()


def test_spawn_cable_authors_newton_import_contract(stage):
    """Test the USD contract consumed by Newton's cable importer."""
    points = [(0.0, 0.0, 0.0), (0.2, 0.1, 0.0), (0.4, 0.0, 0.0)]
    material = CableMaterialCfg(thickness=0.02, density=1234.0, stretch_stiffness=2.5e6, bend_stiffness=7.5e4)
    cfg = CableCfg(positions=points, physics_material=material)

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
    assert not curve_prim.HasAttribute("connections")
    assert not curve_prim.HasAPI(UsdPhysics.CollisionAPI)

    binding_api = UsdShade.MaterialBindingAPI(curve_prim)
    physics_binding = binding_api.GetDirectBinding("physics")
    assert physics_binding.GetMaterialPath() == "/World/Cable/geometry/physics_material"
    assert physics_binding.GetMaterialPurpose() == "physics"

    physics_material_prim = stage.GetPrimAtPath(physics_binding.GetMaterialPath())
    assert physics_material_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert "PhysicsCurvesDeformableMaterialAPI" in physics_material_prim.GetPrimTypeInfo().GetAppliedAPISchemas()


def test_spawn_cable_applies_optional_collision_api(stage):
    """Test optional cable collision authoring."""
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.4, 0.0)],
        physics_material=CableMaterialCfg(),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=False),
    )

    cfg.func("/World/Cable", cfg)
    curve_prim = stage.GetPrimAtPath("/World/Cable/geometry/mesh")

    assert curve_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert curve_prim.GetAttribute("physics:collisionEnabled").Get() is False


def test_spawn_cable_rejects_too_few_points(stage):
    """Test that an open cable requires at least two segments."""
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.0, 0.2, 0.0)],
        physics_material=CableMaterialCfg(),
    )

    with pytest.raises(ValueError, match="at least three"):
        cfg.func("/World/Cable", cfg)

    assert not stage.GetPrimAtPath("/World/Cable").IsValid()
