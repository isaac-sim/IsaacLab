# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from pxr import UsdGeom

pytestmark = pytest.mark.integration


def test_physx_deformable_body_fragment_authors_attrs_via_writer():
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyCfg

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils.string import string_to_callable

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = UsdGeom.Xform.Define(stage, "/World/B").GetPrim()

    cfg = PhysxDeformableBodyCfg(solver_position_iteration_count=32, linear_damping=0.1)
    func = string_to_callable(cfg.func) if isinstance(cfg.func, str) else cfg.func
    ok = func(cfg, "/World/B", stage)

    assert ok is True
    assert "PhysxBaseDeformableBodyAPI" in prim.GetAppliedSchemas()
    assert prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Get() == 32
    assert abs(prim.GetAttribute("physxDeformableBody:linearDamping").Get() - 0.1) < 1e-6


def test_physx_surface_deformable_body_fragment_metadata():
    from isaaclab_physx.sim.schemas import PhysxSurfaceDeformableBodyCfg

    from isaaclab.sim.schemas import DeformableBodyFragment

    cfg = PhysxSurfaceDeformableBodyCfg(collision_pair_update_frequency=2)
    assert isinstance(cfg, DeformableBodyFragment)
    assert type(cfg)._usd_applied_schema == "PhysxSurfaceDeformableBodyAPI"
    assert type(cfg)._deformable_types == ("surface",)
