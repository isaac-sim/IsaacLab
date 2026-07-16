# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from pxr import UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import CableMaterialCfg


def test_spawn_cable_material_authors_canonical_schema():
    sim_utils.create_new_stage()
    cfg = CableMaterialCfg(
        thickness=0.02,
        density=1200.0,
        stretch_stiffness=3.0e6,
        bend_stiffness=0.04,
    )

    prim = cfg.func("/World/Looks/Cable", cfg)

    assert prim.IsA(UsdShade.Material)
    assert bool(UsdPhysics.MaterialAPI(prim))
    assert "PhysicsCurvesDeformableMaterialAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    assert prim.GetAttribute("physics:thickness").Get() == pytest.approx(0.02)
    assert prim.GetAttribute("physics:density").Get() == pytest.approx(1200.0)
    assert prim.GetAttribute("physics:stretchStiffness").Get() == pytest.approx(3.0e6)
    assert prim.GetAttribute("physics:bendStiffness").Get() == pytest.approx(0.04)
