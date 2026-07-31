# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg

# Cable joint degrees of freedom, in the order Newton lays them out.
_STRETCH, _SHEAR, _BEND, _TWIST = range(4)


def _import_cable_joint_stiffness(**material_kwargs) -> list[float]:
    """Spawn a cable, import it into Newton, and return its first joint's per-DOF stiffness."""
    sim_utils.create_new_stage()
    stage = sim_utils.get_current_stage()
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0)],
        physics_material=CableMaterialCfg(
            thickness=0.02, density=1000.0, stretch_stiffness=1.0e6, bend_stiffness=2.0e4, **material_kwargs
        ),
    )
    cfg.func("/World/Cable", cfg)

    builder = newton.ModelBuilder()
    builder.add_usd(stage, root_path="/World/Cable")
    dof0 = builder.joint_qd_start[0]
    return [builder.joint_target_ke[dof0 + offset] for offset in range(4)]


def test_newton_cable_shear_and_twist_fall_back_when_unset():
    """Test that unset shear/twist reuse the stretch/bend stiffness."""
    stiffness = _import_cable_joint_stiffness()

    assert stiffness[_SHEAR] == pytest.approx(stiffness[_STRETCH])
    assert stiffness[_TWIST] == pytest.approx(stiffness[_BEND])


def test_newton_cable_authored_shear_and_twist_override_fallbacks():
    """Test that authored moduli decouple shear from stretch and twist from bend."""
    fallback = _import_cable_joint_stiffness()
    stiffness = _import_cable_joint_stiffness(shear_stiffness=9.0e9, twist_stiffness=7.0e7)

    # Stretch and bend are untouched; only the newly authored degrees of freedom move.
    assert stiffness[_STRETCH] == pytest.approx(fallback[_STRETCH])
    assert stiffness[_BEND] == pytest.approx(fallback[_BEND])
    assert stiffness[_SHEAR] > 100.0 * stiffness[_STRETCH]
    assert stiffness[_TWIST] > 100.0 * stiffness[_BEND]


def test_newton_cable_partial_and_zero_shear_twist():
    """Test that one authored modulus leaves the other's fallback intact and an authored zero is kept."""
    only_shear = _import_cable_joint_stiffness(shear_stiffness=9.0e9)
    assert only_shear[_SHEAR] > 100.0 * only_shear[_STRETCH]
    assert only_shear[_TWIST] == pytest.approx(only_shear[_BEND])

    zero_twist = _import_cable_joint_stiffness(twist_stiffness=0.0)
    assert zero_twist[_TWIST] == pytest.approx(0.0)
    assert zero_twist[_BEND] > 0.0


def test_newton_imports_cable_without_registry():
    """Test that Newton imports the authored cable directly from USD."""
    stage = sim_utils.create_new_stage()
    material = CableMaterialCfg(
        thickness=0.02,
        density=1234.0,
        stretch_stiffness=2.5e6,
        bend_stiffness=7.5e4,
    )
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.1, 0.0), (0.4, 0.0, 0.0)],
        physics_material=material,
    )
    cfg.func("/World/Cable", cfg)

    cable_path = "/World/Cable/geometry/mesh"
    cable_prim = stage.GetPrimAtPath(cable_path)
    assert not cable_prim.HasAttribute("connections")

    builder = newton.ModelBuilder()
    import_result = builder.add_usd(stage, root_path="/World/Cable", return_deformable_results=True)
    cable_attrs = import_result["path_cable_attrs"][cable_path]

    assert builder.body_count == 2
    assert builder.joint_count == 1
    assert builder.shape_count == 2
    assert import_result["path_cable_map"][cable_path] == ([0, 1], [0])
    assert cable_attrs["closed"] is False
    assert cable_attrs["material"]["thickness"] == pytest.approx(material.thickness)
    assert cable_attrs["material"]["density"] == pytest.approx(material.density)
    assert cable_attrs["material"]["stretchStiffness"] == pytest.approx(material.stretch_stiffness)
    assert cable_attrs["material"]["bendStiffness"] == pytest.approx(material.bend_stiffness)
    # Shear/twist are left unauthored when unset.
    assert "shearStiffness" not in cable_attrs["material"]
    assert "twistStiffness" not in cable_attrs["material"]


def test_newton_imports_cable_with_adjacent_collision_filtering():
    """Test that only connected cable segments are collision-filtered."""
    stage = sim_utils.create_new_stage()
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0)],
        physics_material=CableMaterialCfg(),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )
    cfg.func("/World/Cable", cfg)

    builder = newton.ModelBuilder()
    builder.add_usd(stage, root_path="/World/Cable")
    filter_pairs = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}

    assert all(flag & newton.ShapeFlags.COLLIDE_SHAPES for flag in builder.shape_flags)
    assert filter_pairs == {(0, 1), (1, 2)}
