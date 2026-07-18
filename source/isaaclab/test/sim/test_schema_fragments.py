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
# SchemaFragment base, RigidBodyFragment marker, UsdPhysicsRigidBodyCfg
# -------------------------------------------------------------------------------------


def test_fragment_metadata_defaults():
    from isaaclab.sim.schemas import RigidBodyFragment, SchemaFragment, UsdPhysicsRigidBodyCfg

    cfg = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    assert isinstance(cfg, RigidBodyFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physics"
    assert type(cfg)._usd_applied_schema is None  # anchor applies RigidBodyAPI, not the fragment
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.rigid_body_enabled is True and cfg.kinematic_enabled is None


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
# PhysxRigidBodyCfg (isaaclab_physx)
# -------------------------------------------------------------------------------------


def test_physx_rigid_body_fragment_writes_physx_namespace():
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/B2")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    apply_namespaced(PhysxRigidBodyCfg(linear_damping=0.1, disable_gravity=True), "/World/B2", stage)
    assert abs(prim.GetAttribute("physxRigidBody:linearDamping").Get() - 0.1) < 1e-6
    assert prim.GetAttribute("physxRigidBody:disableGravity").Get() is True


# -------------------------------------------------------------------------------------
# MujocoRigidBodyCfg (isaaclab_newton)
# -------------------------------------------------------------------------------------


def test_mujoco_rigid_body_fragment_writes_mjc_namespace():
    from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/B3")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    apply_namespaced(MujocoRigidBodyCfg(gravcomp=1.0), "/World/B3", stage)
    assert abs(prim.GetAttribute("mjc:gravcomp").Get() - 1.0) < 1e-6


def test_mujoco_rigid_body_fragment_does_not_write_gravcomp_when_none():
    # fragment-path equivalent of the legacy test_mujoco_gravcomp_not_written_when_none:
    # an unset gravcomp must not author mjc:gravcomp
    from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/B3b")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    apply_namespaced(MujocoRigidBodyCfg(), "/World/B3b", stage)
    assert prim.GetAttribute("mjc:gravcomp").Get() is None


# -------------------------------------------------------------------------------------
# apply_rigid_body_properties dispatch (implicit anchor + multi-namespace)
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
            PhysxRigidBodyCfg(linear_damping=0.2),
            MujocoRigidBodyCfg(gravcomp=1.0),
        ],
        stage,
    )
    prim = stage.GetPrimAtPath("/World/B4")
    assert bool(UsdPhysics.RigidBodyAPI(prim))  # implicit anchor applied
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert abs(prim.GetAttribute("physxRigidBody:linearDamping").Get() - 0.2) < 1e-6
    assert abs(prim.GetAttribute("mjc:gravcomp").Get() - 1.0) < 1e-6


# -------------------------------------------------------------------------------------
# spawner slot accepts a fragment list + transition routing
# -------------------------------------------------------------------------------------


def test_spawn_shape_with_rigid_fragment_list():
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), PhysxRigidBodyCfg(linear_damping=0.3)],
    )
    cfg.func("/World/Cube", cfg)
    prim = sim_utils.get_current_stage().GetPrimAtPath("/World/Cube")
    assert bool(UsdPhysics.RigidBodyAPI(prim))
    assert abs(prim.GetAttribute("physxRigidBody:linearDamping").Get() - 0.3) < 1e-6


# -------------------------------------------------------------------------------------
# public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg  # noqa: F401
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg  # noqa: F401

    from isaaclab.sim.schemas import (  # noqa: F401
        RigidBodyFragment,
        SchemaFragment,
        UsdPhysicsRigidBodyCfg,
        apply_namespaced,
        apply_rigid_body_properties,
    )


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


def test_apply_rigid_body_properties_raises_on_invalid_prim():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_rigid_body_properties("/World/DoesNotExist", [UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)], stage)


def test_apply_rigid_body_properties_aggregates_fragment_results():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Agg")

    # a fragment whose applier reports failure must make the aggregate return False
    failing = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    failing.func = lambda cfg, prim_path, stage=None: False
    assert apply_rigid_body_properties("/World/Agg", [failing], stage) is False

    # all-succeeding fragments return True
    ok = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    assert apply_rigid_body_properties("/World/Agg", [ok], stage) is True


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
