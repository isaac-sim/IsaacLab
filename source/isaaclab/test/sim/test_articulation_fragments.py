# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os

import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

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


def test_apply_articulation_root_properties_processes_siblings_and_aggregates_results():
    """Every non-nested sibling root is visited and a failing applier makes the family result False."""
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    for path in ("/World/A", "/World/B", "/World/A/Nested"):
        root = _make_xform(stage, path)
        UsdPhysics.ArticulationRootAPI.Apply(root)

    visited = []

    def record_result(_cfg, path, _stage):
        visited.append(path)
        return path != "/World/B"

    result = apply_articulation_root_properties("/World", [PhysxArticulationCfg(func=record_result)], stage)

    assert visited == ["/World/A", "/World/B"]
    assert result is False


def test_apply_articulation_root_properties_does_not_duplicate_instance_proxy_root(caplog):
    """A root hidden in an instance suppresses define-fresh but is skipped because proxies are read-only."""
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    source = _make_xform(stage, "/World/Source")
    source_base = _make_xform(stage, "/World/Source/base")
    UsdPhysics.ArticulationRootAPI.Apply(source_base)
    instance = _make_xform(stage, "/World/Asset")
    instance.GetReferences().AddInternalReference(source.GetPath())
    instance.SetInstanceable(True)
    proxy_root = stage.GetPrimAtPath("/World/Asset/base")
    assert proxy_root.IsInstanceProxy() and proxy_root.HasAPI(UsdPhysics.ArticulationRootAPI)

    with caplog.at_level("WARNING"):
        result = apply_articulation_root_properties(
            "/World/Asset", [PhysxArticulationCfg(solver_position_iteration_count=4)], stage
        )

    assert result is False
    assert not instance.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert proxy_root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not proxy_root.GetAttribute("physxArticulation:solverPositionIterationCount").HasAuthoredValue()
    assert "/World/Asset/base" in caplog.text


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


def test_apply_articulation_root_properties_enables_existing_joint_and_relocates_root():
    """An existing joint is enabled, but PhysX root placement is still normalized through the manager."""
    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    parent = _make_xform(stage, "/World/ExistingJointRobot")
    root = _make_xform(stage, "/World/ExistingJointRobot/base")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/ExistingJointRobot/base/FixedJoint")
    joint.CreateBody1Rel().SetTargets([root.GetPath()])
    joint.CreateJointEnabledAttr(False)

    apply_articulation_root_properties("/World/ExistingJointRobot", [], stage, fix_root_link=True)

    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    assert joints == [joint.GetPrim()]
    assert joint.GetJointEnabledAttr().Get() is True
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)


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


def test_apply_articulation_root_properties_fix_root_link_without_active_simulation_raises(monkeypatch):
    """Creating a fixed root joint with no active ``SimulationContext`` raises a clear error: the
    backend that authors (and possibly relocates) the root is resolved from the live simulation, so
    without one core cannot fix the base on its own.
    """
    from isaaclab.sim import SimulationContext as _SimCtx
    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    root = _make_xform(stage, "/World/Robot3")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # simulate "no active simulation" so the backend cannot be resolved
    monkeypatch.setattr(_SimCtx, "instance", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError):
        apply_articulation_root_properties("/World/Robot3", [], stage, fix_root_link=True)


def test_physx_and_newton_fragments_fix_root_link_keeps_single_root():
    """Composing a PhysX and a Newton fragment with ``fix_root_link=True`` must leave exactly one
    articulation root, with *every* backend's schema on that resulting root.

    Regression: PhysX relocates the root to the parent when fixing the base. Writing fragments after
    the relocation lands both the PhysX and Newton schemas on the parent; the former child (root link)
    must not retain ``NewtonArticulationRootAPI`` / ``newton:*`` (which would leave a stray second root
    because that API composes ``PhysicsArticulationRootAPI``).
    """
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Robot5")
    child = _make_xform(stage, "/World/Robot5/base")
    UsdPhysics.RigidBodyAPI.Apply(child)
    UsdPhysics.ArticulationRootAPI.Apply(child)

    apply_articulation_root_properties(
        "/World/Robot5",
        [
            PhysxArticulationCfg(solver_position_iteration_count=8),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        stage,
        fix_root_link=True,
    )

    parent = stage.GetPrimAtPath("/World/Robot5")
    # exactly one articulation root remains, and it is the (relocated) parent
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1 and roots[0] == parent
    # every backend's schema is authored on that single resulting root ...
    assert parent.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    assert parent.GetAttribute("newton:selfCollisionEnabled").Get() is True
    # ... and nothing is stranded on the former child root link
    assert not child.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not child.GetAttribute("newton:selfCollisionEnabled").HasAuthoredValue()


def test_physx_fix_root_link_migrates_preauthored_newton_root_api():
    """A pre-authored backend root API (as on URDF/MJCF-imported assets) moves with the root when
    PhysX relocates it, together with its authored attributes.

    Regression: ``NewtonArticulationRootAPI`` composes ``PhysicsArticulationRootAPI``, so removing
    only the directly applied anchor from the former root link leaves that prim an articulation root
    through schema composition -- two roots -- with the ``newton:*`` values stranded on the wrong prim.
    """
    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/UrdfBot")
    child = _make_xform(stage, "/World/UrdfBot/base")
    UsdPhysics.RigidBodyAPI.Apply(child)
    # the asset ships with the Newton root API (and its attribute) already authored on the root link
    child.AddAppliedSchema("NewtonArticulationRootAPI")
    child.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    # precondition: the composed schema makes the prim an articulation root without a direct anchor
    assert child.HasAPI(UsdPhysics.ArticulationRootAPI)

    apply_articulation_root_properties("/World/UrdfBot", [], stage, fix_root_link=True)

    parent = stage.GetPrimAtPath("/World/UrdfBot")
    # exactly one articulation root remains, and it is the (relocated) parent
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1 and roots[0] == parent
    # the pre-authored backend schema and its value moved with the root
    assert "NewtonArticulationRootAPI" in parent.GetAppliedSchemas()
    assert parent.GetAttribute("newton:selfCollisionEnabled").Get() is True
    # the former root link no longer carries the composed root API
    assert not child.HasAPI(UsdPhysics.ArticulationRootAPI)


def test_physx_fix_root_link_preserves_complete_authored_property_spec():
    """Relocation preserves sampled, metadata, and connection opinions rather than one default value."""
    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/UsdBot")
    child = _make_xform(stage, "/World/UsdBot/base")
    UsdPhysics.RigidBodyAPI.Apply(child)
    UsdPhysics.MassAPI.Apply(child)
    UsdPhysics.ArticulationRootAPI.Apply(child)
    child.AddAppliedSchema("PhysxArticulationAPI")
    driver = _make_xform(stage, "/World/Driver")
    driver_attr = driver.CreateAttribute("output", Sdf.ValueTypeNames.Float)
    driver_attr.Set(0.5)
    source_attr = child.GetAttribute("physxArticulation:sleepThreshold")
    source_attr.Set(0.1, Usd.TimeCode(1.0))
    source_attr.Set(0.2, Usd.TimeCode(2.0))
    source_attr.SetMetadata("documentation", "sampled sleep threshold")
    source_attr.AddConnection(driver_attr.GetPath())

    apply_articulation_root_properties("/World/UsdBot", [], stage, fix_root_link=True)

    parent = stage.GetPrimAtPath("/World/UsdBot")
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "PhysxArticulationAPI" in parent.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert not child.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "PhysxArticulationAPI" not in child.GetPrimTypeInfo().GetAppliedAPISchemas()
    moved_attr = parent.GetAttribute("physxArticulation:sleepThreshold")
    assert not stage.GetRootLayer().GetAttributeAtPath(moved_attr.GetPath()).HasInfo("default")
    assert moved_attr.GetTimeSamples() == [1.0, 2.0]
    assert moved_attr.Get(Usd.TimeCode(1.0)) == pytest.approx(0.1)
    assert moved_attr.Get(Usd.TimeCode(2.0)) == pytest.approx(0.2)
    assert moved_attr.GetMetadata("documentation") == "sampled sleep threshold"
    assert moved_attr.GetConnections() == [driver_attr.GetPath()]
    assert child.HasAPI(UsdPhysics.RigidBodyAPI) and child.HasAPI(UsdPhysics.MassAPI)
    assert not parent.HasAPI(UsdPhysics.RigidBodyAPI) and not parent.HasAPI(UsdPhysics.MassAPI)


def test_apply_articulation_root_properties_rejects_non_fragment_items():
    """A list containing a non-fragment (e.g. a legacy single cfg) raises a clear ``TypeError`` --
    not an ``AttributeError`` deep inside fragment dispatch."""
    from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/BadList")
    with pytest.raises(TypeError, match="ArticulationRootFragment"):
        apply_articulation_root_properties(
            "/World/BadList",
            [PhysxArticulationRootPropertiesCfg(solver_position_iteration_count=8)],
            stage,
        )


def test_apply_articulation_root_properties_topology_only_does_not_stamp_root():
    """A topology-only call (empty fragments) on a prim with no articulation root anywhere must not
    apply ``UsdPhysics.ArticulationRootAPI`` as a side effect -- the anchor is presence-gated on
    fragments. The topology flag is ignored with a warning instead."""
    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/NoRoot")
    UsdPhysics.RigidBodyAPI.Apply(prim)

    apply_articulation_root_properties("/World/NoRoot", [], stage, fix_root_link=False)

    assert not prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())


def test_apply_articulation_root_properties_honors_explicit_stage():
    """Fragment writes and fixed-joint lookup both stay on the explicitly supplied stage."""
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    current_stage = sim_utils.get_current_stage()
    current_root = _make_xform(current_stage, "/World/AltStageRoot")
    UsdPhysics.ArticulationRootAPI.Apply(current_root)
    current_joint = UsdPhysics.FixedJoint.Define(current_stage, "/World/AltStageRoot/FixedJoint")
    current_joint.CreateBody1Rel().SetTargets([current_root.GetPath()])
    current_joint.CreateJointEnabledAttr(True)

    other_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(other_stage, "/World")
    other_root = UsdGeom.Xform.Define(other_stage, "/World/AltStageRoot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(other_root)
    other_joint = UsdPhysics.FixedJoint.Define(other_stage, "/World/AltStageRoot/FixedJoint")
    other_joint.CreateBody1Rel().SetTargets([other_root.GetPath()])
    other_joint.CreateJointEnabledAttr(True)

    apply_articulation_root_properties(
        "/World/AltStageRoot",
        [PhysxArticulationCfg(solver_position_iteration_count=4)],
        other_stage,
        fix_root_link=False,
    )

    assert other_root.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 4
    assert other_joint.GetJointEnabledAttr().Get() is False
    assert not current_root.GetAttribute("physxArticulation:solverPositionIterationCount").HasAuthoredValue()
    assert current_joint.GetJointEnabledAttr().Get() is True


# -------------------------------------------------------------------------------------
# fix_articulation_root is a PhysicsManager capability (resolved by cfg.physics.class_type)
# -------------------------------------------------------------------------------------


def test_base_manager_fix_articulation_root_is_world_anchored_and_idempotent():
    """The neutral capability pins the current pose and enables rather than duplicates an existing joint."""
    from isaaclab.physics import PhysicsManager

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/BaseRobot")
    root = _make_xform(stage, "/World/BaseRobot/base")
    UsdGeom.Xform(root).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    result = PhysicsManager.fix_articulation_root(root, stage)
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    assert result == root and len(joints) == 1
    joint = UsdPhysics.FixedJoint(joints[0])
    assert list(joint.GetBody1Rel().GetTargets()) == [root.GetPath()]
    assert joint.GetPrim().GetAttribute("physics:localPos0").Get() == Gf.Vec3f(1.0, 2.0, 3.0)
    assert not stage.GetPrimAtPath("/World/BaseRobot").HasAPI(UsdPhysics.ArticulationRootAPI)

    joint.CreateJointEnabledAttr(False)
    assert PhysicsManager.fix_articulation_root(root, stage) == root
    assert joint.GetJointEnabledAttr().Get() is True
    assert len([prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]) == 1


def test_base_manager_fix_articulation_root_requires_rigid_body():
    """The capability raises NotImplementedError on a non-rigid-body root (cannot anchor the joint)."""
    from isaaclab.physics import PhysicsManager

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    root = _make_xform(stage, "/World/BaseNoRB")
    UsdPhysics.ArticulationRootAPI.Apply(root)  # articulation root but NOT a rigid body
    with pytest.raises(NotImplementedError):
        PhysicsManager.fix_articulation_root(root, stage)


def test_articulation_fragment_and_legacy_cfg_match_physx_schema():
    """Both interfaces cover exactly all six attributes registered by PhysxArticulationAPI."""
    import dataclasses

    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxArticulationRootPropertiesCfg

    from pxr import PhysxSchema

    from isaaclab.utils.string import to_camel_case

    def fields(cls):
        return {field.name for field in dataclasses.fields(cls) if field.name != "func"}

    schema_attrs = {name.split(":", 1)[1] for name in PhysxSchema.PhysxArticulationAPI.GetSchemaAttributeNames()}
    fragment_attrs = {to_camel_case(name, "cC") for name in fields(PhysxArticulationCfg)}
    legacy_attrs = {
        to_camel_case(name, "cC") for name in fields(PhysxArticulationRootPropertiesCfg) - {"fix_root_link"}
    }
    assert fragment_attrs == schema_attrs
    assert legacy_attrs == schema_attrs


def test_newton_legacy_cfg_matches_equivalent_fragment_composition():
    """A Newton legacy subclass still routes its inherited articulation_enabled base field."""
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg, NewtonArticulationRootPropertiesCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties, modify_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    legacy = _make_xform(stage, "/World/LegacyNewton")
    fragments = _make_xform(stage, "/World/FragmentNewton")
    UsdPhysics.ArticulationRootAPI.Apply(legacy)
    UsdPhysics.ArticulationRootAPI.Apply(fragments)

    modify_articulation_root_properties(
        legacy.GetPath(),
        NewtonArticulationRootPropertiesCfg(articulation_enabled=False, self_collision_enabled=True),
        stage,
    )
    apply_articulation_root_properties(
        fragments.GetPath(),
        [
            PhysxArticulationCfg(articulation_enabled=False),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        stage,
    )

    for prim in (legacy, fragments):
        assert prim.GetAttribute("physxArticulation:articulationEnabled").Get() is False
        assert prim.GetAttribute("newton:selfCollisionEnabled").Get() is True


# -------------------------------------------------------------------------------------
# end-to-end: the from-files transition bridge routes articulation_props by type
# -------------------------------------------------------------------------------------


def _author_articulation_usd(path: str) -> None:
    """Author a minimal USD asset: a parent Xform with a rigid-body child carrying the articulation root."""
    asset_stage = Usd.Stage.CreateNew(path)
    robot = UsdGeom.Xform.Define(asset_stage, "/Robot")
    base = UsdGeom.Xform.Define(asset_stage, "/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    asset_stage.SetDefaultPrim(robot.GetPrim())
    asset_stage.Save()


@pytest.mark.parametrize("articulation_props", [None, []], ids=["none", "empty"])
def test_spawn_from_usd_file_topology_only_honors_fix_root_link(tmp_path, articulation_props):
    """None and an empty fragment collection both honor the independent topology flag."""
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    usd_path = os.path.join(tmp_path, "articulation.usda")
    _author_articulation_usd(usd_path)

    cfg = UsdFileCfg(usd_path=usd_path, articulation_props=articulation_props, fix_root_link=True)
    _spawn_from_usd_file("/World/FromUsdTopology", usd_path, cfg)

    stage = sim_utils.get_current_stage()
    assert any(prim.IsA(UsdPhysics.FixedJoint) for prim in stage.Traverse())
    assert stage.GetPrimAtPath("/World/FromUsdTopology").HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/World/FromUsdTopology/base").HasAPI(UsdPhysics.ArticulationRootAPI)


def test_spawn_from_usd_file_applies_composed_fragment_list(tmp_path):
    """The real FileCfg transition bridge composes backend fragments on the relocated root."""
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    usd_path = os.path.join(tmp_path, "fragment_articulation.usda")
    _author_articulation_usd(usd_path)
    cfg = UsdFileCfg(
        usd_path=usd_path,
        articulation_props=[
            PhysxArticulationCfg(solver_position_iteration_count=8),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        fix_root_link=True,
    )

    _spawn_from_usd_file("/World/FromUsdFragments", usd_path, cfg)

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/FromUsdFragments")
    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [root]
    assert root.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    assert root.GetAttribute("newton:selfCollisionEnabled").Get() is True


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
    )
