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


_INVALID_CABLE_MATERIAL_VALUES = [
    ("thickness", 0.0),
    ("density", 0.0),
    ("stretch_stiffness", -1.0),
    ("bend_stiffness", -1.0),
    ("shear_stiffness", -1.0),
    ("twist_stiffness", -1.0),
    ("shear_stiffness", float("nan")),
    ("twist_stiffness", float("inf")),
    ("thickness", float("nan")),
]


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
    assert not sim_utils.has_deformable_body_api(curve_prim)
    assert not curve_prim.HasAPI(UsdPhysics.CollisionAPI)

    physics_binding = UsdShade.MaterialBindingAPI(curve_prim).GetDirectBinding("physics")
    assert physics_binding.GetMaterialPath() == "/World/Cable/geometry/physics_material"
    assert physics_binding.GetMaterialPurpose() == "physics"

    physics_material_prim = stage.GetPrimAtPath(physics_binding.GetMaterialPath())
    assert "PhysicsCurvesDeformableMaterialAPI" in physics_material_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert physics_material_prim.GetAttribute("physics:thickness").Get() == pytest.approx(0.001)
    assert physics_material_prim.GetAttribute("physics:density").Get() == pytest.approx(1000.0)
    assert physics_material_prim.GetAttribute("physics:stretchStiffness").Get() == pytest.approx(1.0e9)
    assert physics_material_prim.GetAttribute("physics:bendStiffness").Get() == pytest.approx(1.0e6)
    # Shear/twist are left unauthored when unset.
    assert not physics_material_prim.HasAttribute("physics:shearStiffness")
    assert not physics_material_prim.HasAttribute("physics:twistStiffness")


def test_spawn_cable_authors_optional_shear_and_twist_stiffness(stage):
    cfg = CableCfg(
        positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        physics_material=CableMaterialCfg(shear_stiffness=2.5e8, twist_stiffness=0.0),
    )

    cfg.func("/World/Cable", cfg)
    material_prim = stage.GetPrimAtPath("/World/Cable/geometry/physics_material")

    assert material_prim.GetAttribute("physics:shearStiffness").Get() == pytest.approx(2.5e8)
    # An authored zero is kept as zero rather than treated as unset.
    assert material_prim.GetAttribute("physics:twistStiffness").Get() == pytest.approx(0.0)


def test_spawn_cable_authors_optional_collision_properties(stage):
    cfg = CableCfg(
        positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        physics_material=CableMaterialCfg(),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )

    cfg.func("/World/Cable", cfg)
    curve_prim = stage.GetPrimAtPath("/World/Cable/geometry/mesh")

    assert curve_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.CollisionAPI(curve_prim).GetCollisionEnabledAttr().Get() is True


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), "at least three"),
        (((0.0, 0.0, 0.0), (1.0e-8, 0.0, 0.0), (0.1, 0.0, 0.0)), "separated"),
        (((0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)), "exactly three coordinates"),
        (((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (0.2, 0.0, 0.0)), "finite coordinates"),
    ],
)
def test_spawn_cable_rejects_invalid_positions_without_authoring(stage, positions, message):
    cfg = CableCfg(positions=positions, physics_material=CableMaterialCfg())

    with pytest.raises(ValueError, match=message):
        cfg.func("/World/Cable", cfg)

    assert not stage.GetPrimAtPath("/World/Cable").IsValid()


def test_spawn_cable_rejects_invalid_material_without_authoring(stage):
    material = CableMaterialCfg(thickness=0.0)
    cfg = CableCfg(
        positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        physics_material=material,
    )

    with pytest.raises(ValueError, match="thickness"):
        cfg.func("/World/Cable", cfg)

    assert not stage.GetPrimAtPath("/World/Cable").IsValid()


@pytest.mark.parametrize(("field", "value"), _INVALID_CABLE_MATERIAL_VALUES)
def test_spawn_cable_material_rejects_invalid_values_without_authoring(stage, field, value):
    material = CableMaterialCfg()
    setattr(material, field, value)

    with pytest.raises(ValueError, match=field):
        material.func("/World/CableMaterial", material)

    assert not stage.GetPrimAtPath("/World/CableMaterial").IsValid()
