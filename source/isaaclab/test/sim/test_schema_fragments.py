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


def _make_xform(stage, path="/World/Body"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# apply_namespaced generic applier
# -------------------------------------------------------------------------------------


def test_apply_namespaced_writes_only_set_fields():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    apply_namespaced(UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), "/World/Body", stage)
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    # ``kinematicEnabled`` is a RigidBodyAPI fallback attr (so HasAttribute is True), but the
    # None field must not be authored by apply_namespaced.
    assert not prim.GetAttribute("physics:kinematicEnabled").HasAuthoredValue()


# -------------------------------------------------------------------------------------
# apply_rigid_body_properties dispatch (explicit anchor creation + multi-namespace)
# -------------------------------------------------------------------------------------


def test_apply_rigid_body_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/B4")
    apply_rigid_body_properties(
        "/World/B4",
        [
            UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(linear_damping=0.2, disable_gravity=True),
            MujocoRigidBodyCfg(gravcomp=1.0),
        ],
        create_if_missing=True,
        stage=stage,
    )
    prim = stage.GetPrimAtPath("/World/B4")
    assert bool(UsdPhysics.RigidBodyAPI(prim))  # anchor created on the bare prim
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert abs(prim.GetAttribute("physxRigidBody:linearDamping").Get() - 0.2) < 1e-6
    assert prim.GetAttribute("physxRigidBody:disableGravity").Get() is True
    assert abs(prim.GetAttribute("mjc:gravcomp").Get() - 1.0) < 1e-6


# -------------------------------------------------------------------------------------
# Review follow-ups -- prim-validity guard, aggregated return, namespace invariant guard
# -------------------------------------------------------------------------------------


def test_apply_namespaced_raises_on_invalid_prim():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    # no prim authored at this path -> GetPrimAtPath returns an invalid prim
    with pytest.raises(ValueError):
        apply_namespaced(UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), "/World/DoesNotExist", stage)


@pytest.mark.parametrize("family", ["rigid_body", "mass"])
def test_apply_family_properties_aggregates_fragment_results(family):
    from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg, apply_mass_properties, apply_rigid_body_properties

    writer, make_fragment = {
        "rigid_body": (apply_rigid_body_properties, lambda: UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)),
        "mass": (apply_mass_properties, lambda: MassCfg(mass=1.0)),
    }[family]

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Agg")

    # a fragment whose applier reports failure must make the aggregate return False
    failing = make_fragment()
    failing.func = lambda cfg, prim_path, stage=None: False
    assert writer("/World/Agg", [failing], create_if_missing=True, stage=stage) is False

    # all-succeeding fragments return True
    ok = make_fragment()
    assert writer("/World/Agg", [ok], create_if_missing=True, stage=stage) is True


def test_apply_namespaced_raises_without_namespace():
    from typing import ClassVar

    from isaaclab.sim.schemas import RigidBodyFragment, apply_namespaced
    from isaaclab.utils import configclass

    @configclass
    class _NoNamespaceFragment(RigidBodyFragment):
        # deliberately leaves ``_usd_namespace`` as None, violating the fragment invariant that
        # every field is authored as a namespaced USD attribute
        _usd_namespace: ClassVar[str | None] = None
        rigid_body_enabled: bool | None = None

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/NoNs")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    with pytest.raises(ValueError):
        apply_namespaced(_NoNamespaceFragment(rigid_body_enabled=True), "/World/NoNs", stage)


def test_fragment_mapping_normalizes_bare_fragment_and_list():
    """A bare fragment (or list) on a spawner field is shorthand for the anchor-prim mapping."""
    from isaaclab.sim.schemas import MassCfg, MassPropertiesCfg, UsdPhysicsRigidBodyCfg
    from isaaclab.sim.spawners._utils import fragment_mapping

    frag = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    assert fragment_mapping(frag) == {"": [frag]}

    a, b = MassCfg(mass=1.0), MassCfg(density=10.0)
    assert fragment_mapping([a, b]) == {"": [a, b]}
    assert fragment_mapping((a, b)) == {"": [a, b]}

    # an explicit mapping is passed through untouched
    mapping = {"/.*": [frag]}
    assert fragment_mapping(mapping) is mapping

    # legacy dataclass cfgs report None so callers route them to the legacy writers
    assert fragment_mapping(MassPropertiesCfg(mass=1.0)) is None
    assert fragment_mapping(None) is None


def test_shape_spawner_accepts_bare_fragment_for_props():
    """A bare fragment authors on the shape's anchor prim, exactly as ``{"": [...]}`` would."""
    from pxr import UsdPhysics

    from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    cfg = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
        mass_props=MassCfg(mass=0.5),
    )
    cfg.func("/World/Bare", cfg)

    prim = stage.GetPrimAtPath("/World/Bare")
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert abs(prim.GetAttribute("physics:mass").Get() - 0.5) < 1e-6
