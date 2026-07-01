# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import pytest

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _make_revolute_joint(stage, path="/World/Articulation/joint_0"):
    UsdGeom.Xform.Define(stage, "/World/Articulation")
    UsdGeom.Cube.Define(stage, "/World/Articulation/body0")
    UsdGeom.Cube.Define(stage, "/World/Articulation/body1")
    UsdPhysics.RevoluteJoint.Define(stage, path)
    return stage.GetPrimAtPath(path)


def _make_prismatic_joint(stage, path="/World/Articulation/joint_p"):
    UsdGeom.Xform.Define(stage, "/World/Articulation")
    UsdGeom.Cube.Define(stage, "/World/Articulation/body0")
    UsdGeom.Cube.Define(stage, "/World/Articulation/body1")
    UsdPhysics.PrismaticJoint.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# Fragment metadata
# -------------------------------------------------------------------------------------


def test_drive_fragment_metadata_defaults():
    from isaaclab.sim.schemas import JointDriveFragment, SchemaFragment, UsdPhysicsDriveCfg

    cfg = UsdPhysicsDriveCfg(drive_type="acceleration", max_force=80.0, stiffness=10.0, damping=0.1)
    assert isinstance(cfg, JointDriveFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace is None  # typed multi-instance DriveAPI, no namespace writes
    assert type(cfg)._usd_applied_schema is None  # DriveAPI applied by apply_drive (presence-gated)
    assert cfg.func == "isaaclab.sim.schemas:apply_drive"
    assert cfg.stiffness == 10.0 and cfg.damping == 0.1


def test_drive_fragment_max_effort_alias():
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg

    with pytest.warns(DeprecationWarning, match="max_effort"):
        cfg = UsdPhysicsDriveCfg(max_effort=42.0)
    assert cfg.max_force == 42.0
    assert cfg.max_effort is None


# -------------------------------------------------------------------------------------
# apply_drive -- revolute (angular) rad->deg conversion
# -------------------------------------------------------------------------------------


def test_apply_drive_revolute_converts_rad_to_deg():
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_drive

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    assert apply_drive(
        UsdPhysicsDriveCfg(drive_type="acceleration", max_force=80.0, stiffness=10.0, damping=0.1),
        prim.GetPath().pathString,
        stage,
    )
    assert bool(UsdPhysics.DriveAPI(prim, "angular"))
    assert prim.GetAttribute("drive:angular:physics:type").Get() == "acceleration"
    assert prim.GetAttribute("drive:angular:physics:maxForce").Get() == pytest.approx(80.0, rel=1e-6)
    # angular stiffness/damping are converted from radian to degree units
    assert prim.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(10.0 * math.pi / 180.0, rel=1e-6)
    assert prim.GetAttribute("drive:angular:physics:damping").Get() == pytest.approx(0.1 * math.pi / 180.0, rel=1e-6)


# -------------------------------------------------------------------------------------
# apply_drive -- prismatic (linear) no conversion
# -------------------------------------------------------------------------------------


def test_apply_drive_prismatic_writes_linear_unchanged():
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_drive

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prismatic_joint(stage)
    assert apply_drive(
        UsdPhysicsDriveCfg(drive_type="force", max_force=42.0, stiffness=10.0, damping=0.1),
        prim.GetPath().pathString,
        stage,
    )
    assert bool(UsdPhysics.DriveAPI(prim, "linear"))
    assert prim.GetAttribute("drive:linear:physics:type").Get() == "force"
    assert prim.GetAttribute("drive:linear:physics:maxForce").Get() == pytest.approx(42.0, rel=1e-6)
    # linear drives are written as authored (no rad->deg conversion)
    assert prim.GetAttribute("drive:linear:physics:stiffness").Get() == pytest.approx(10.0, rel=1e-6)
    assert prim.GetAttribute("drive:linear:physics:damping").Get() == pytest.approx(0.1, rel=1e-6)


def test_apply_drive_returns_false_on_non_joint():
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_drive

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/NotAJoint")
    assert apply_drive(UsdPhysicsDriveCfg(stiffness=1.0), "/World/NotAJoint", stage) is False


# -------------------------------------------------------------------------------------
# PhysxJointCfg (isaaclab_physx) -- physxJoint namespace via apply_namespaced
# -------------------------------------------------------------------------------------


def test_physx_joint_fragment_converts_max_velocity_by_joint_type():
    from isaaclab_physx.sim.schemas import PhysxJointCfg, apply_physx_joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    # angular (revolute) joint: rad/s -> deg/s conversion
    rev = _make_revolute_joint(stage)
    apply_physx_joint(PhysxJointCfg(max_joint_velocity=10.0), rev.GetPath().pathString, stage)
    assert rev.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(10.0 * 180.0 / math.pi, rel=1e-6)
    # linear (prismatic) joint: written unchanged
    prismatic = _make_prismatic_joint(stage)
    apply_physx_joint(PhysxJointCfg(max_joint_velocity=10.0), prismatic.GetPath().pathString, stage)
    assert prismatic.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(10.0, rel=1e-6)


def test_physx_joint_fragment_max_velocity_alias():
    from isaaclab_physx.sim.schemas import PhysxJointCfg

    with pytest.warns(DeprecationWarning, match="max_velocity"):
        cfg = PhysxJointCfg(max_velocity=10.0)
    assert cfg.max_joint_velocity == 10.0
    assert cfg.max_velocity is None


# -------------------------------------------------------------------------------------
# MujocoJointCfg (isaaclab_newton) -- mjc namespace via apply_namespaced
# -------------------------------------------------------------------------------------


def test_mujoco_joint_fragment_writes_mjc_namespace():
    from isaaclab_newton.sim.schemas import MujocoJointCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    apply_namespaced(MujocoJointCfg(actuatorgravcomp=True), prim.GetPath().pathString, stage)
    assert prim.GetAttribute("mjc:actuatorgravcomp").Get() is True


def test_mujoco_joint_applier_does_not_write_actuatorgravcomp_when_none():
    # fragment-path equivalent of the legacy test_mujoco_actuatorgravcomp_not_written_when_none:
    # an unset actuatorgravcomp must not *author* mjc:actuatorgravcomp through the joint applier.
    # (Unlike the legacy path, the fragment applies its MjcJointAPI schema, so the attribute exists
    # and resolves to the schema default ``False`` -- but it must not be authored to ``True``.)
    from isaaclab_newton.sim.schemas import MujocoJointCfg, apply_mujoco_joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    apply_mujoco_joint(MujocoJointCfg(), prim.GetPath().pathString, stage)
    assert not prim.GetAttribute("mjc:actuatorgravcomp").HasAuthoredValue()


def test_mujoco_joint_actuatorgravcomp_enables_child_body_gravcomp():
    # actuatorgravcomp is inert unless the actuated body has non-zero mjc:gravcomp, so the Mujoco
    # joint applier flips gravcomp on the joint's child body (physics:body1) when it is unset
    from isaaclab_newton.sim.schemas import MujocoJointCfg

    from isaaclab.sim.schemas import apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    UsdPhysics.RevoluteJoint(prim).CreateBody1Rel().SetTargets(["/World/Articulation/body1"])
    apply_joint_drive_properties("/World/Articulation", [MujocoJointCfg(actuatorgravcomp=True)], stage)
    body = stage.GetPrimAtPath("/World/Articulation/body1")
    assert body.GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)


def test_mujoco_joint_without_actuatorgravcomp_leaves_body_gravcomp_untouched():
    # the gravcomp coupling must fire ONLY when actuatorgravcomp is requested; an unset flag leaves
    # the child body's mjc:gravcomp unauthored
    from isaaclab_newton.sim.schemas import MujocoJointCfg

    from isaaclab.sim.schemas import apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    UsdPhysics.RevoluteJoint(prim).CreateBody1Rel().SetTargets(["/World/Articulation/body1"])
    apply_joint_drive_properties("/World/Articulation", [MujocoJointCfg()], stage)
    body = stage.GetPrimAtPath("/World/Articulation/body1")
    assert body.GetAttribute("mjc:gravcomp").Get() is None


def test_mujoco_joint_actuatorgravcomp_enables_gravcomp_on_every_joint_body():
    # per-joint dispatch must enable gravcomp on each joint's own child body across the articulation
    from isaaclab_newton.sim.schemas import MujocoJointCfg

    from isaaclab.sim.schemas import apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Articulation")
    for link in ("link_a", "link_b"):
        UsdGeom.Cube.Define(stage, f"/World/Articulation/{link}")
    j0 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/joint_0")
    j0.CreateBody1Rel().SetTargets(["/World/Articulation/link_a"])
    j1 = UsdPhysics.PrismaticJoint.Define(stage, "/World/Articulation/joint_1")
    j1.CreateBody1Rel().SetTargets(["/World/Articulation/link_b"])
    apply_joint_drive_properties("/World/Articulation", [MujocoJointCfg(actuatorgravcomp=True)], stage)
    # both the revolute joint's body and the prismatic joint's body get gravcomp enabled
    assert stage.GetPrimAtPath("/World/Articulation/link_a").GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)
    assert stage.GetPrimAtPath("/World/Articulation/link_b").GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)


def test_mujoco_joint_actuatorgravcomp_preserves_authored_body_gravcomp():
    # an explicitly authored body gravcomp must not be clobbered by the actuatorgravcomp auto-enable
    from isaaclab_newton.sim.schemas import MujocoJointCfg

    from isaaclab.sim.schemas import apply_joint_drive_properties
    from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    UsdPhysics.RevoluteJoint(prim).CreateBody1Rel().SetTargets(["/World/Articulation/body1"])
    body = stage.GetPrimAtPath("/World/Articulation/body1")
    safe_set_attribute_on_usd_prim(body, "mjc:gravcomp", 0.5, camel_case=False)
    apply_joint_drive_properties("/World/Articulation", [MujocoJointCfg(actuatorgravcomp=True)], stage)
    assert body.GetAttribute("mjc:gravcomp").Get() == pytest.approx(0.5)


# -------------------------------------------------------------------------------------
# apply_joint_drive_properties dispatch (presence-gated DriveAPI + multi-namespace)
# -------------------------------------------------------------------------------------


def test_apply_joint_drive_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import MujocoJointCfg
    from isaaclab_physx.sim.schemas import PhysxJointCfg

    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    apply_joint_drive_properties(
        "/World/Articulation",
        [
            UsdPhysicsDriveCfg(drive_type="acceleration", max_force=80.0, stiffness=10.0, damping=0.1),
            PhysxJointCfg(max_joint_velocity=5.0),
            MujocoJointCfg(actuatorgravcomp=True),
        ],
        stage,
    )
    assert bool(UsdPhysics.DriveAPI(prim, "angular"))  # presence-gated anchor applied
    assert prim.GetAttribute("drive:angular:physics:maxForce").Get() == pytest.approx(80.0, rel=1e-6)
    assert prim.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(10.0 * math.pi / 180.0, rel=1e-6)
    # revolute joint -> rad/s to deg/s conversion via apply_physx_joint
    assert prim.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(5.0 * 180.0 / math.pi, rel=1e-6)
    assert prim.GetAttribute("mjc:actuatorgravcomp").Get() is True


def test_apply_joint_drive_properties_without_drive_does_not_apply_drive_api():
    from isaaclab_physx.sim.schemas import PhysxJointCfg

    from isaaclab.sim.schemas import apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    apply_joint_drive_properties("/World/Articulation", [PhysxJointCfg(max_joint_velocity=5.0)], stage)
    # DriveAPI is presence-gated: not applied when no UsdPhysicsDriveCfg fragment is present
    assert not bool(UsdPhysics.DriveAPI(prim, "angular"))
    # revolute joint -> rad/s to deg/s conversion via apply_physx_joint
    assert prim.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(5.0 * 180.0 / math.pi, rel=1e-6)


def test_apply_joint_drive_properties_skips_tendon_child_joint():
    """A tendon-child joint (``PhysxTendonAxisAPI`` without the root API) must be skipped wholesale
    by the dispatch loop: no fragment -- drive, physxJoint, or mjc -- may author on it, matching the
    legacy :func:`modify_joint_drive_properties` writer (which skipped the whole prim)."""
    from isaaclab_physx.sim.schemas import PhysxJointCfg

    from pxr import PhysxSchema

    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    joint = _make_revolute_joint(stage)
    PhysxSchema.PhysxTendonAxisAPI.Apply(joint, "axis0")  # tendon child: axis API, no root API
    applied = str(joint.GetAppliedSchemas())
    assert "PhysxTendonAxisAPI" in applied and "PhysxTendonAxisRootAPI" not in applied

    apply_joint_drive_properties(
        "/World/Articulation",
        [
            UsdPhysicsDriveCfg(drive_type="acceleration", max_force=80.0, stiffness=10.0, damping=0.1),
            PhysxJointCfg(max_joint_velocity=5.0),
        ],
        stage,
    )
    # neither the presence-gated DriveAPI nor the physxJoint fragment may author on a tendon child
    assert not bool(UsdPhysics.DriveAPI(joint, "angular"))
    assert not joint.GetAttribute("physxJoint:maxJointVelocity").HasAuthoredValue()


def test_apply_joint_drive_properties_skips_joint_via_registered_predicate(monkeypatch):
    """Core delegates joint exclusion to backend-registered predicates: one returning True skips the
    joint, with no backend-specific schema knowledge in core.

    monkeypatch clears the module-global predicate list (other tests register the PhysX detector
    session-wide) and restores it afterwards, keeping this isolated.
    """
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, _backend_hooks, apply_joint_drive_properties

    monkeypatch.setattr(_backend_hooks, "_JOINT_DRIVE_SKIP_PREDICATES", [])
    _backend_hooks.register_joint_drive_skip_predicate(lambda prim: True)  # exclude every joint

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    joint = _make_revolute_joint(stage)
    apply_joint_drive_properties("/World/Articulation", [UsdPhysicsDriveCfg(stiffness=10.0)], stage)
    assert not bool(UsdPhysics.DriveAPI(joint, "angular"))


def test_apply_joint_drive_properties_authors_when_no_skip_predicate(monkeypatch):
    """Empty predicate registry (the default) skips nothing -- the writer authors on the joint."""
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, _backend_hooks, apply_joint_drive_properties

    monkeypatch.setattr(_backend_hooks, "_JOINT_DRIVE_SKIP_PREDICATES", [])

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    joint = _make_revolute_joint(stage)
    apply_joint_drive_properties("/World/Articulation", [UsdPhysicsDriveCfg(stiffness=10.0)], stage)
    assert bool(UsdPhysics.DriveAPI(joint, "angular"))


def test_apply_joint_drive_properties_ensure_drives_exist_seeds_stiffness():
    from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_joint_drive_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_revolute_joint(stage)
    # neither stiffness nor damping authored -> ensure_drives_exist seeds a minimal stiffness
    apply_joint_drive_properties(
        "/World/Articulation", [UsdPhysicsDriveCfg(max_force=1.0)], stage, ensure_drives_exist=True
    )
    assert bool(UsdPhysics.DriveAPI(prim, "angular"))
    assert prim.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(1e-3 * math.pi / 180.0, rel=1e-6)


# -------------------------------------------------------------------------------------
# public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_newton.sim.schemas import MujocoJointCfg  # noqa: F401
    from isaaclab_physx.sim.schemas import PhysxJointCfg  # noqa: F401

    from isaaclab.sim.schemas import (  # noqa: F401
        JointDriveFragment,
        SchemaFragment,
        UsdPhysicsDriveCfg,
        apply_drive,
        apply_joint_drive_properties,
    )
