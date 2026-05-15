# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.shapes import CableCfg

pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture
def sim():
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.1))
    sim_utils.update_stage()
    yield sim
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim.clear_instance()


def _basic_material() -> NewtonCableMaterialCfg:
    return NewtonCableMaterialCfg(
        stretch_stiffness=1.0e9,
        bend_stiffness=1.0e-3,
        density=1500.0,
    )


def test_spawn_cable(sim):
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        width=0.01,
        physics_material=_basic_material(),
    )
    prim = cfg.func("/World/Cable", cfg)
    assert prim.IsValid()

    curve_prim = sim.stage.GetPrimAtPath("/World/Cable/geometry/mesh")
    assert curve_prim.IsValid()
    curves = UsdGeom.BasisCurves(curve_prim)
    points = list(curves.GetPointsAttr().Get())
    assert len(points) == 3
    counts = list(curves.GetCurveVertexCountsAttr().Get())
    assert counts == [3]
    widths = list(curves.GetWidthsAttr().Get())
    assert widths == pytest.approx([0.01, 0.01, 0.01])  # cfg.width, broadcast per control point
    assert curves.GetTypeAttr().Get() == "linear"


def test_spawn_cable_validation_wrong_material(sim):
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        width=0.01,
        physics_material=RigidBodyMaterialCfg(),
    )
    with pytest.raises(ValueError, match="NewtonCableMaterialCfg"):
        cfg.func("/World/Cable", cfg)


def test_spawn_cable_validation_rigid_props_rejected(sim):
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        width=0.01,
        physics_material=_basic_material(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )
    with pytest.raises(ValueError, match="rigid_props"):
        cfg.func("/World/Cable", cfg)


def test_spawn_cable_authors_newton_material_attrs(sim):
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        width=0.01,
        physics_material=_basic_material(),
    )
    cfg.func("/World/Cable", cfg)
    mat_prim = sim.stage.GetPrimAtPath("/World/Cable/geometry/physics_material")
    assert mat_prim.IsValid()
    # Material fields land under newton:* namespace (camelCase).
    attr = mat_prim.GetAttribute("newton:stretchStiffness")
    assert attr.IsValid()
    assert attr.Get() == pytest.approx(1.0e9)


def test_spawn_cable_authors_visual_and_physics_at_distinct_paths(sim):
    """When both visual and physics materials are configured, they author at
    distinct sub-paths and neither overwrites the other."""
    from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        width=0.01,
        physics_material=_basic_material(),
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    )
    cfg.func("/World/Cable", cfg)
    assert sim.stage.GetPrimAtPath("/World/Cable/geometry/physics_material").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cable/geometry/visual_material").IsValid()
