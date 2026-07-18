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


def _make_xform(stage, path="/World/Body"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# Fragment metadata -- MassFragment marker, MassCfg
# -------------------------------------------------------------------------------------


def test_fragment_metadata_defaults():
    from isaaclab.sim.schemas import MassCfg, MassFragment, SchemaFragment

    cfg = MassCfg(mass=2.0)
    assert isinstance(cfg, MassFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physics"
    assert type(cfg)._usd_applied_schema is None  # anchor applies MassAPI, not the fragment
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.mass == 2.0 and cfg.density is None


# -------------------------------------------------------------------------------------
# apply_namespaced writes only the set fields under the physics namespace
# -------------------------------------------------------------------------------------


def test_apply_namespaced_writes_only_set_fields():
    from isaaclab.sim.schemas import MassCfg, apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage)
    UsdPhysics.MassAPI.Apply(prim)
    apply_namespaced(MassCfg(mass=3.0), "/World/Body", stage)
    assert abs(prim.GetAttribute("physics:mass").Get() - 3.0) < 1e-6
    # None field must not be authored (density exists as a MassAPI fallback attr)
    assert not prim.GetAttribute("physics:density").HasAuthoredValue()


# -------------------------------------------------------------------------------------
# apply_mass_properties dispatch (implicit MassAPI anchor)
# -------------------------------------------------------------------------------------


def test_apply_mass_properties_applies_anchor_and_writes_fields():
    from isaaclab.sim.schemas import MassCfg, apply_mass_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/B2")
    apply_mass_properties("/World/B2", [MassCfg(mass=5.0, density=100.0)], stage)
    prim = stage.GetPrimAtPath("/World/B2")
    assert bool(UsdPhysics.MassAPI(prim))  # implicit anchor applied
    assert abs(prim.GetAttribute("physics:mass").Get() - 5.0) < 1e-6
    assert abs(prim.GetAttribute("physics:density").Get() - 100.0) < 1e-6


# -------------------------------------------------------------------------------------
# spawner slot accepts a fragment list + transition routing
# -------------------------------------------------------------------------------------


def test_spawn_shape_with_mass_fragment_list():
    from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)],
        mass_props=[MassCfg(mass=4.0)],
    )
    cfg.func("/World/Cube", cfg)
    prim = sim_utils.get_current_stage().GetPrimAtPath("/World/Cube")
    assert bool(UsdPhysics.MassAPI(prim))
    assert abs(prim.GetAttribute("physics:mass").Get() - 4.0) < 1e-6


def test_spawn_shape_with_single_mass_fragment():
    # the ``mass_props`` slot advertises a single fragment (convenience form), not only a list;
    # the spawn shim must route a bare fragment through ``apply_mass_properties`` (not the legacy writer)
    from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)],
        mass_props=MassCfg(mass=4.0),
    )
    cfg.func("/World/CubeSingle", cfg)
    prim = sim_utils.get_current_stage().GetPrimAtPath("/World/CubeSingle")
    assert bool(UsdPhysics.MassAPI(prim))
    assert abs(prim.GetAttribute("physics:mass").Get() - 4.0) < 1e-6


# -------------------------------------------------------------------------------------
# Review follow-ups -- prim-validity guard, aggregated return, empty-list no-op
# -------------------------------------------------------------------------------------


def test_apply_mass_properties_raises_on_invalid_prim():
    from isaaclab.sim.schemas import MassCfg, apply_mass_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    # no prim authored at this path -> GetPrimAtPath returns an invalid prim
    with pytest.raises(ValueError):
        apply_mass_properties("/World/DoesNotExist", [MassCfg(mass=1.0)], stage)


def test_apply_mass_properties_aggregates_fragment_results():
    from isaaclab.sim.schemas import MassCfg, apply_mass_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Agg")

    # a fragment whose applier reports failure must make the aggregate return False
    failing = MassCfg(mass=1.0)
    failing.func = lambda cfg, prim_path, stage=None: False
    assert apply_mass_properties("/World/Agg", [failing], stage) is False

    # all-succeeding fragments return True
    ok = MassCfg(mass=1.0)
    assert apply_mass_properties("/World/Agg", [ok], stage) is True


def test_spawn_shape_with_empty_mass_list_is_noop():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)],
        mass_props=[],
    )
    # an empty fragment list routes through the fragment path and applies nothing (no exception)
    cfg.func("/World/Cube", cfg)
    prim = sim_utils.get_current_stage().GetPrimAtPath("/World/Cube")
    # mass anchor is not required when there are zero fragments to apply
    assert not prim.GetAttribute("physics:mass").HasAuthoredValue()


# -------------------------------------------------------------------------------------
# public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab.sim.schemas import (  # noqa: F401
        MassCfg,
        MassFragment,
        SchemaFragment,
        apply_mass_properties,
    )
