# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg


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
