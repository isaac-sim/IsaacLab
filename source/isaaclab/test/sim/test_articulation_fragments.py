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

from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _make_xform(stage, path="/World/Art"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# ArticulationRootFragment marker + metadata
# -------------------------------------------------------------------------------------


def test_articulation_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import ArticulationRootFragment, SchemaFragment

    cfg = PhysxArticulationCfg(articulation_enabled=True)
    assert isinstance(cfg, ArticulationRootFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physxArticulation"
    assert type(cfg)._usd_applied_schema == "PhysxArticulationAPI"
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.articulation_enabled is True and cfg.enabled_self_collisions is None


# -------------------------------------------------------------------------------------
# PhysxArticulationCfg writes physxArticulation:* namespace
# -------------------------------------------------------------------------------------


def test_physx_articulation_fragment_writes_physx_namespace():
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/A1")
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    apply_namespaced(
        PhysxArticulationCfg(articulation_enabled=True, enabled_self_collisions=False, sleep_threshold=0.1),
        "/World/A1",
        stage,
    )
    assert prim.GetAttribute("physxArticulation:articulationEnabled").Get() is True
    assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is False
    assert abs(prim.GetAttribute("physxArticulation:sleepThreshold").Get() - 0.1) < 1e-6


# -------------------------------------------------------------------------------------
# NewtonArticulationCfg writes newton:* namespace
# -------------------------------------------------------------------------------------


def test_newton_articulation_fragment_writes_newton_namespace():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/A2")
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    apply_namespaced(NewtonArticulationCfg(self_collision_enabled=True), "/World/A2", stage)
    assert prim.GetAttribute("newton:selfCollisionEnabled").Get() is True


# -------------------------------------------------------------------------------------
# apply_articulation_root_properties dispatch (anchor + multi-namespace composition)
# -------------------------------------------------------------------------------------


def test_apply_articulation_root_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/A3")
    apply_articulation_root_properties(
        "/World/A3",
        [
            PhysxArticulationCfg(enabled_self_collisions=True, solver_position_iteration_count=8),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        stage,
    )
    prim = stage.GetPrimAtPath("/World/A3")
    assert bool(UsdPhysics.ArticulationRootAPI(prim))  # presence-gated anchor applied
    assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert prim.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    assert prim.GetAttribute("newton:selfCollisionEnabled").Get() is True


# -------------------------------------------------------------------------------------
# Regression: root on a CHILD prim (USD assets) must be tuned in place, not duplicated
# -------------------------------------------------------------------------------------


def test_apply_articulation_root_properties_tunes_existing_child_root():
    """When the articulation root lives on a child prim (as in USD assets), the writer must tune
    that existing root rather than stamp a second ``ArticulationRootAPI`` on the input (top) prim
    -- a duplicate root mis-writes the properties and violates the 'exactly one root' invariant.
    """
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    top = _make_xform(stage, "/World/Asset")
    child = _make_xform(stage, "/World/Asset/base")
    UsdPhysics.ArticulationRootAPI.Apply(child)  # asset already carries its root on a child prim

    apply_articulation_root_properties(
        "/World/Asset",
        [PhysxArticulationCfg(solver_position_iteration_count=8)],
        stage,
    )

    # the existing child root is tuned ...
    assert child.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert child.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    # ... and NO duplicate root / stray write is added on the top prim
    assert not top.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not top.GetAttribute("physxArticulation:solverPositionIterationCount").HasAuthoredValue()
    # exactly one ArticulationRootAPI exists in the subtree
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1 and roots[0] == child


# -------------------------------------------------------------------------------------
# fix_root_link spawner-level flag: toggles an existing fixed joint
# -------------------------------------------------------------------------------------


def test_apply_articulation_root_properties_toggles_existing_fixed_joint():
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    # root prim with a rigid body and an articulation root
    root = _make_xform(stage, "/World/A4")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # author an existing global fixed joint between the world and the root link
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/A4/FixedJoint")
    joint.CreateBody1Rel().SetTargets(["/World/A4"])
    joint.CreateJointEnabledAttr(True)

    apply_articulation_root_properties(
        "/World/A4",
        [PhysxArticulationCfg(articulation_enabled=True)],
        stage,
        fix_root_link=False,
    )
    assert joint.GetJointEnabledAttr().Get() is False


# -------------------------------------------------------------------------------------
# fix_root_link spawner-level flag: creates a fixed joint and reparents the root
# -------------------------------------------------------------------------------------


def test_apply_articulation_root_properties_creates_fixed_joint_and_reparents_root():
    """fix_root_link=True with no existing fixed joint: a fixed joint is created and the
    articulation root is moved from the rigid-body root link to its parent (PhysX parser
    limitation -- a fixed joint on a rigid body is otherwise treated as a maximal-coordinate tree).
    """
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    # parent xform + a rigid-body root link carrying the articulation root
    _make_xform(stage, "/World/Robot")
    root = _make_xform(stage, "/World/Robot/base")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # no existing global fixed joint -> the writer must create one
    assert not any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())

    apply_articulation_root_properties(
        "/World/Robot",
        [PhysxArticulationCfg(articulation_enabled=True)],
        stage,
        fix_root_link=True,
    )

    parent = stage.GetPrimAtPath("/World/Robot")
    # a fixed joint was created ...
    assert any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    # ... and the articulation root was moved from the root link to its parent
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)


def test_apply_articulation_root_properties_fix_root_link_requires_rigid_body():
    """fix_root_link=True on a non-rigid-body root raises NotImplementedError: the writer cannot
    determine the first rigid body link to anchor the fixed joint to."""
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    root = _make_xform(stage, "/World/Robot2")
    UsdPhysics.ArticulationRootAPI.Apply(root)  # articulation root but NOT a rigid body
    with pytest.raises(NotImplementedError):
        apply_articulation_root_properties(
            "/World/Robot2",
            [PhysxArticulationCfg(articulation_enabled=True)],
            stage,
            fix_root_link=True,
        )


def test_apply_articulation_root_properties_fix_root_link_without_backend_raises(monkeypatch):
    """Creating a fixed root joint with no registered backend creator raises a clear error: the
    backend-specific creation logic lives in the backends, so core cannot fix the base on its own.

    Uses monkeypatch to clear the module-global creator registry (other tests/backends register
    session-wide); monkeypatch restores it afterwards so this stays isolated.
    """
    from isaaclab.sim.schemas import _backend_hooks, apply_articulation_root_properties

    monkeypatch.setattr(_backend_hooks, "_FIXED_ROOT_JOINT_CREATORS", {})

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    root = _make_xform(stage, "/World/Robot3")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    with pytest.raises(RuntimeError):
        apply_articulation_root_properties("/World/Robot3", [], stage, fix_root_link=True)


def test_register_fixed_root_joint_creator_selects_active_backend(monkeypatch):
    """The writer selects the creator registered for the active simulation's ``cfg.physics`` type and
    ignores creators registered for other backend cfg types -- no probing of inactive backends."""
    from isaaclab.sim.schemas import _backend_hooks, apply_articulation_root_properties

    monkeypatch.setattr(_backend_hooks, "_FIXED_ROOT_JOINT_CREATORS", {})
    called = {}

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    active_cfg_type = type(SimulationContext.instance().cfg.physics)

    class _OtherBackendCfg:
        """Stand-in for an inactive backend's physics-cfg type."""

    # a creator keyed by an inactive backend's cfg type must be ignored; the active one's is used
    _backend_hooks.register_fixed_root_joint_creator(
        _OtherBackendCfg, lambda p, s: called.__setitem__("path", "INACTIVE")
    )
    _backend_hooks.register_fixed_root_joint_creator(
        active_cfg_type, lambda p, s: called.__setitem__("path", p.GetPath().pathString)
    )

    stage = sim_utils.get_current_stage()
    root = _make_xform(stage, "/World/Robot4")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    apply_articulation_root_properties("/World/Robot4", [], stage, fix_root_link=True)
    assert called["path"] == "/World/Robot4"


# -------------------------------------------------------------------------------------
# create_fixed_root_joint -- backend-neutral, pure-USD world<->prim fixed joint
# -------------------------------------------------------------------------------------


def test_create_fixed_root_joint_authors_world_anchored_joint():
    """create_fixed_root_joint authors a world<->prim ``UsdPhysics.FixedJoint``: ``body1`` targets the
    prim, ``body0`` is left empty (the world frame), and the body1-side local frame is identity."""
    from isaaclab.sim.schemas import create_fixed_root_joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/Fix1")
    joint_prim = create_fixed_root_joint(prim, stage)

    assert joint_prim.IsA(UsdPhysics.FixedJoint)
    joint = UsdPhysics.FixedJoint(joint_prim)
    assert list(joint.GetBody1Rel().GetTargets()) == [prim.GetPath()]
    # body0 left empty -> the world frame
    body0_rel = joint.GetBody0Rel()
    assert not body0_rel or list(body0_rel.GetTargets()) == []
    # the body1-side local frame is identity
    assert joint_prim.GetAttribute("physics:localPos1").Get() == Gf.Vec3f(0.0)
    assert joint_prim.GetAttribute("physics:localRot1").Get() == Gf.Quatf(1.0)


def test_create_fixed_root_joint_pins_prim_at_current_world_pose():
    """The world-side local frame is set to the prim's current world transform, so the constraint pins
    the body where it is rather than teleporting it to the world origin."""
    from isaaclab.sim.schemas import create_fixed_root_joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/Fix2")
    UsdGeom.Xform(prim).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))

    joint_prim = create_fixed_root_joint(prim, stage)
    pos0 = joint_prim.GetAttribute("physics:localPos0").Get()
    assert (pos0[0], pos0[1], pos0[2]) == pytest.approx((1.0, 2.0, 3.0))


def test_create_fixed_root_joint_uses_unique_names():
    """Repeated calls author distinct joint prims (no path collision)."""
    from isaaclab.sim.schemas import create_fixed_root_joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/Fix3")
    first = create_fixed_root_joint(prim, stage)
    second = create_fixed_root_joint(prim, stage)
    assert first.GetPath() != second.GetPath()
    assert first.IsA(UsdPhysics.FixedJoint) and second.IsA(UsdPhysics.FixedJoint)


# -------------------------------------------------------------------------------------
# active-backend detection: creator is resolved by the live cfg.physics type
# -------------------------------------------------------------------------------------


def test_resolve_fixed_root_joint_creator_matches_cfg_type(monkeypatch):
    """_resolve_fixed_root_joint_creator returns the creator registered for the given physics-cfg type,
    and None for an unregistered type or no active simulation -- a pure dict lookup, no probing."""
    from isaaclab.sim.schemas import _backend_hooks

    monkeypatch.setattr(_backend_hooks, "_FIXED_ROOT_JOINT_CREATORS", {})

    class _CfgA:
        pass

    class _CfgB:
        pass

    def _creator_a(prim, stage):
        pass

    _backend_hooks.register_fixed_root_joint_creator(_CfgA, _creator_a)

    assert _backend_hooks._resolve_fixed_root_joint_creator(_CfgA) is _creator_a
    assert _backend_hooks._resolve_fixed_root_joint_creator(_CfgB) is None
    assert _backend_hooks._resolve_fixed_root_joint_creator(None) is None


def test_newton_creator_authors_joint_without_reparenting():
    """The Newton fixed-root-joint creator authors a ``UsdPhysics.FixedJoint`` but, unlike PhysX, leaves
    the articulation root in place (Newton reads the fixed joint directly), and pulls in no PhysX deps."""
    from isaaclab_newton.sim.schemas import _create_fixed_root_joint_newton

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/NewtonRobot")
    root = _make_xform(stage, "/World/NewtonRobot/base")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    _create_fixed_root_joint_newton(root, stage)

    assert any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    # Newton does NOT relocate the articulation root to the parent (that is a PhysX-parser workaround)
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/World/NewtonRobot").HasAPI(UsdPhysics.ArticulationRootAPI)


def test_newton_creator_requires_rigid_body():
    """The Newton creator raises NotImplementedError on a non-rigid-body root (cannot anchor the joint)."""
    from isaaclab_newton.sim.schemas import _create_fixed_root_joint_newton

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    root = _make_xform(stage, "/World/NewtonNoRB")
    UsdPhysics.ArticulationRootAPI.Apply(root)  # articulation root but NOT a rigid body
    with pytest.raises(NotImplementedError):
        _create_fixed_root_joint_newton(root, stage)


# -------------------------------------------------------------------------------------
# public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg  # noqa: F401
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg  # noqa: F401

    from isaaclab.sim.schemas import (  # noqa: F401
        ArticulationRootFragment,
        SchemaFragment,
        apply_articulation_root_properties,
        create_fixed_root_joint,
    )
