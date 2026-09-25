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

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Release the simulation context after each test."""
    yield
    SimulationContext.clear_instance()


# -------------------------------------------------------------------------------------
# apply_mass_properties creation and empty-list no-op
# -------------------------------------------------------------------------------------


def test_apply_mass_properties_creates_on_every_matched_prim():
    """Creation applies ``MassAPI`` to every matched prim lacking it, rigid body or not."""
    from isaaclab.sim.schemas import MassCfg, apply_mass_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, "/World/Bot").GetPrim()
    body = UsdGeom.Xform.Define(stage, "/World/Bot/link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    plain = UsdGeom.Xform.Define(stage, "/World/Bot/frame").GetPrim()

    result = apply_mass_properties(
        "/World/Bot(/.*)?", [MassCfg(mass=2.0, density=100.0)], create_if_missing=True, stage=stage
    )

    assert result is True
    for prim in (root, body, plain):
        assert prim.HasAPI(UsdPhysics.MassAPI), prim.GetPath()
        assert prim.GetAttribute("physics:mass").Get() == pytest.approx(2.0), prim.GetPath()
        assert prim.GetAttribute("physics:density").Get() == pytest.approx(100.0), prim.GetPath()


def test_spawn_shape_with_empty_mass_list_is_noop():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        rigid_props={"": [UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)]},
        mass_props={"": []},
    )
    # an entry with an empty fragment list routes through the fragment path and applies nothing (no exception)
    cfg.func("/World/Cube", cfg)
    prim = sim_utils.get_current_stage().GetPrimAtPath("/World/Cube")
    # mass anchor is not required when there are zero fragments to apply
    assert not prim.GetAttribute("physics:mass").HasAuthoredValue()
