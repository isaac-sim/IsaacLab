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
    ("thickness", -1.0),
    ("thickness", float("nan")),
    ("thickness", float("inf")),
    ("density", 0.0),
    ("density", -1.0),
    ("density", float("nan")),
    ("density", float("inf")),
    ("stretch_stiffness", -1.0),
    ("stretch_stiffness", float("nan")),
    ("stretch_stiffness", float("inf")),
    ("bend_stiffness", -1.0),
    ("bend_stiffness", float("nan")),
    ("bend_stiffness", float("inf")),
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
    assert physics_material_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert "PhysicsCurvesDeformableMaterialAPI" in physics_material_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert physics_material_prim.GetAttribute("physics:thickness").Get() == pytest.approx(0.001)
    assert physics_material_prim.GetAttribute("physics:density").Get() == pytest.approx(1000.0)
    assert physics_material_prim.GetAttribute("physics:stretchStiffness").Get() == pytest.approx(1.0e9)
    assert physics_material_prim.GetAttribute("physics:bendStiffness").Get() == pytest.approx(1.0e6)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), "at least three"),
        (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), "separated"),
        (((0.0, 0.0, 0.0), (1.0e-8, 0.0, 0.0), (0.1, 0.0, 0.0)), "separated"),
        (((0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)), "exactly three coordinates"),
        (((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (0.2, 0.0, 0.0)), "finite coordinates"),
        (((0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0), (0.2, 0.0, 0.0)), "finite coordinates"),
    ],
)
def test_spawn_cable_rejects_invalid_positions_without_authoring(stage, positions, message):
    cfg = CableCfg(positions=positions, physics_material=CableMaterialCfg())

    with pytest.raises(ValueError, match=message):
        cfg.func("/World/Cable", cfg)

    assert not stage.GetPrimAtPath("/World/Cable").IsValid()


@pytest.mark.parametrize("field", ["mass_props", "rigid_props", "collision_props", "activate_contact_sensors"])
def test_cable_cfg_rejects_rigid_only_fields(field):
    kwargs = {field: False if field == "activate_contact_sensors" else None}

    with pytest.raises(TypeError, match=field):
        CableCfg(
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
            physics_material=CableMaterialCfg(),
            **kwargs,
        )


@pytest.mark.parametrize(("field", "value"), _INVALID_CABLE_MATERIAL_VALUES)
def test_spawn_cable_rejects_invalid_material_without_authoring(stage, field, value):
    material = CableMaterialCfg()
    setattr(material, field, value)
    cfg = CableCfg(
        positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        physics_material=material,
    )

    with pytest.raises(ValueError, match=field):
        cfg.func("/World/Cable", cfg)

    assert not stage.GetPrimAtPath("/World/Cable").IsValid()


@pytest.mark.parametrize(("field", "value"), _INVALID_CABLE_MATERIAL_VALUES)
def test_spawn_cable_material_rejects_invalid_values_without_authoring(stage, field, value):
    material = CableMaterialCfg()
    setattr(material, field, value)

    with pytest.raises(ValueError, match=field):
        material.func("/World/CableMaterial", material)

    assert not stage.GetPrimAtPath("/World/CableMaterial").IsValid()
