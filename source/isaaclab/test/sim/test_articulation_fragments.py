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

from pxr import Gf, Usd, UsdGeom, UsdPhysics

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


def test_apply_articulation_root_properties_honors_explicit_stage():
    """The writer authors to the explicitly supplied stage even when it is not the current stage --
    every lookup (root resolution, fixed-joint search) is scoped to that stage."""
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    from isaaclab.sim.schemas import apply_articulation_root_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    # an in-memory stage that is deliberately NOT the current stage
    other_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(other_stage, "/World")
    root = UsdGeom.Xform.Define(other_stage, "/World/AltStageRoot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)

    apply_articulation_root_properties(
        "/World/AltStageRoot",
        [PhysxArticulationCfg(solver_position_iteration_count=4)],
        other_stage,
    )

    # authored on the supplied stage ...
    assert root.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 4
    # ... and not leaked onto the current stage
    assert not sim_utils.get_current_stage().GetPrimAtPath("/World/AltStageRoot").IsValid()


# -------------------------------------------------------------------------------------
# create_fixed_root_joint -- backend-neutral, pure-USD world<->prim fixed joint
# -------------------------------------------------------------------------------------


def test_create_fixed_root_joint_authors_world_anchored_joint():
    """create_fixed_root_joint authors a world<->prim ``UsdPhysics.FixedJoint``: ``body1`` targets the
    prim, ``body0`` is left empty (the world frame), and the body1-side local frame is identity."""
    from isaaclab.sim.utils import create_fixed_root_joint

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
    from isaaclab.sim.utils import create_fixed_root_joint

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
    from isaaclab.sim.utils import create_fixed_root_joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/Fix3")
    first = create_fixed_root_joint(prim, stage)
    second = create_fixed_root_joint(prim, stage)
    assert first.GetPath() != second.GetPath()
    assert first.IsA(UsdPhysics.FixedJoint) and second.IsA(UsdPhysics.FixedJoint)


# -------------------------------------------------------------------------------------
# fix_articulation_root is a PhysicsManager capability (resolved by cfg.physics.class_type)
# -------------------------------------------------------------------------------------


def test_base_manager_fix_articulation_root_authors_joint_without_reparenting():
    """The base :class:`~isaaclab.physics.PhysicsManager` capability authors a ``UsdPhysics.FixedJoint``
    and leaves the articulation root in place, returning the same prim.

    This is the default inherited by backends whose parser reads a fixed joint directly (e.g. Newton,
    OVPhysx) and by any manager subclass that does not override it -- no PhysX-style relocation.
    """
    from isaaclab.physics import PhysicsManager

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/BaseRobot")
    root = _make_xform(stage, "/World/BaseRobot/base")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    result = PhysicsManager.fix_articulation_root(root, stage)

    assert any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    # base default returns the same prim and does NOT relocate the root to the parent
    assert result == root
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/World/BaseRobot").HasAPI(UsdPhysics.ArticulationRootAPI)


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


def test_fix_articulation_root_capability_inherited_by_manager_subclass():
    """A backend manager subclass inherits its parent backend's ``fix_articulation_root`` through the
    normal method-resolution order.

    This is what makes the capability robust where the previous cfg-type registry was not: manager
    dispatch is resolved from ``cfg.physics.class_type``, so a subclassed cfg (e.g. an in-tree
    ``DeformableNewtonCfg(NewtonCfg)``) whose ``class_type`` is an unchanged or subclassed manager
    inherits the correct behaviour, rather than missing an exact-type registry key.
    """
    from isaaclab.physics import PhysicsManager

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))

    # a minimal concrete backend that OVERRIDES the capability (as PhysX does, relocating the root) ...
    class _BackendLike(PhysicsManager):
        @classmethod
        def fix_articulation_root(cls, articulation_prim, stage=None):
            from isaaclab.sim.utils import create_fixed_root_joint

            cls._require_rigid_body_root(articulation_prim)
            create_fixed_root_joint(articulation_prim, stage)
            parent = articulation_prim.GetParent()
            UsdPhysics.ArticulationRootAPI.Apply(parent)
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            return parent

    # ... and a subclass that does not override it (stand-in for a subclassed cfg's class_type)
    class _SubBackend(_BackendLike):
        pass

    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/SubRobot")
    root = _make_xform(stage, "/World/SubRobot/base")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    result = _SubBackend.fix_articulation_root(root, stage)

    parent = stage.GetPrimAtPath("/World/SubRobot")
    # the subclass inherited the overriding backend's behaviour through the MRO
    assert any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    assert result == parent
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)

    # and the real active backend (PhysX) genuinely overrides the base -- so backends specialise via
    # this same inheritance mechanism rather than a parallel registry
    active_manager = SimulationContext.instance().physics_manager
    assert active_manager.fix_articulation_root.__func__ is not PhysicsManager.fix_articulation_root.__func__


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


def test_spawn_from_usd_file_empty_articulation_props_with_fix_root_link(tmp_path):
    """An empty ``articulation_props=[]`` on a USD spawn is a valid (topology-only) fragment collection:
    it must route to the fragment writer and honor ``fix_root_link`` -- not crash in the legacy writer.

    Regression: the legacy path called ``dataclasses.fields([])`` on the empty list and raised
    ``TypeError``, so the direct-writer and spawner entry points disagreed on whether ``[]`` was valid.
    """
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    usd_path = os.path.join(tmp_path, "articulation.usda")
    _author_articulation_usd(usd_path)

    cfg = UsdFileCfg(usd_path=usd_path, articulation_props=[], fix_root_link=True)
    _spawn_from_usd_file("/World/FromUsdA", usd_path, cfg)

    stage = sim_utils.get_current_stage()
    # the topology flag was honored: a fixed joint was created and the root relocated to the parent
    assert any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    parent = stage.GetPrimAtPath("/World/FromUsdA")
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/World/FromUsdA/base").HasAPI(UsdPhysics.ArticulationRootAPI)


def test_spawn_from_usd_file_none_articulation_props_honors_fix_root_link(tmp_path):
    """``fix_root_link`` is a spawner-level flag processed independently of ``articulation_props``:
    with ``articulation_props=None`` it must still fix the base (previously it was silently ignored)."""
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    usd_path = os.path.join(tmp_path, "articulation.usda")
    _author_articulation_usd(usd_path)

    cfg = UsdFileCfg(usd_path=usd_path, articulation_props=None, fix_root_link=True)
    _spawn_from_usd_file("/World/FromUsdB", usd_path, cfg)

    stage = sim_utils.get_current_stage()
    assert any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())


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
    from isaaclab.sim.utils import create_fixed_root_joint  # noqa: F401
