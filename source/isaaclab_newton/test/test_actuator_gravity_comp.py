# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for gravity compensation USD attributes and their propagation to MJCF."""

# pyright: reportPrivateUsage=false

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import tempfile
import xml.etree.ElementTree as ET

import newton
import numpy as np
import pytest
from isaaclab_newton.assets.articulation.articulation import Articulation
from newton.solvers import SolverMuJoCo

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture
def setup_simulation():
    """Fixture to set up and tear down the simulation context."""
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.1))
    yield sim
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim.clear_instance()


def _build_articulation_with_joints(stage):
    """Create a minimal articulation on stage with three revolute joints."""
    sim_utils.create_prim("/World/Robot", prim_type="Xform")
    sim_utils.create_prim("/World/Robot/base", prim_type="Cube")
    sim_utils.create_prim("/World/Robot/link1", prim_type="Cube")
    sim_utils.create_prim("/World/Robot/link2", prim_type="Cube")
    sim_utils.create_prim("/World/Robot/link3", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/link1/shoulder")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/link2/elbow")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/link3/wrist")


# -------------------------------------------------------------------
# Tests: IsaacLab -> USD (_write_actuator_gravity_comp_to_usd)
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_gravity_comp_written_to_matching_joints(setup_simulation):
    """Test that mjc:actuatorgravcomp is set only on joints matching actuators with gravity_compensation=True."""
    stage = sim_utils.get_current_stage()
    _build_articulation_with_joints(stage)

    artic = object.__new__(Articulation)
    artic.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder", "elbow"],
                gravity_compensation=True,
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["wrist"],
                gravity_compensation=None,
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
        },
    )

    artic._write_actuator_gravity_comp_to_usd()

    for joint_path in ["/World/Robot/link1/shoulder", "/World/Robot/link2/elbow"]:
        prim = stage.GetPrimAtPath(joint_path)
        attr = prim.GetAttribute("mjc:actuatorgravcomp")
        assert attr.IsValid(), f"mjc:actuatorgravcomp not set on {joint_path}"
        assert attr.Get() is True, f"mjc:actuatorgravcomp should be True on {joint_path}"

    wrist_prim = stage.GetPrimAtPath("/World/Robot/link3/wrist")
    wrist_attr = wrist_prim.GetAttribute("mjc:actuatorgravcomp")
    assert not wrist_attr.IsValid(), "mjc:actuatorgravcomp should not be set on wrist"


@pytest.mark.isaacsim_ci
def test_gravity_comp_not_written_when_no_actuators_enabled(setup_simulation):
    """Test that no attributes are written when no actuators have gravity_compensation=True."""
    stage = sim_utils.get_current_stage()
    _build_articulation_with_joints(stage)

    artic = object.__new__(Articulation)
    artic.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder", "elbow"],
                gravity_compensation=None,
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
        },
    )

    artic._write_actuator_gravity_comp_to_usd()

    for joint_path in ["/World/Robot/link1/shoulder", "/World/Robot/link2/elbow", "/World/Robot/link3/wrist"]:
        prim = stage.GetPrimAtPath(joint_path)
        attr = prim.GetAttribute("mjc:actuatorgravcomp")
        assert not attr.IsValid(), f"mjc:actuatorgravcomp should not be set on {joint_path}"


@pytest.mark.isaacsim_ci
def test_gravity_comp_with_regex_pattern(setup_simulation):
    """Test that regex patterns in joint_names_expr are matched correctly."""
    stage = sim_utils.get_current_stage()
    _build_articulation_with_joints(stage)

    artic = object.__new__(Articulation)
    artic.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                gravity_compensation=True,
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
        },
    )

    artic._write_actuator_gravity_comp_to_usd()

    for joint_path in ["/World/Robot/link1/shoulder", "/World/Robot/link2/elbow", "/World/Robot/link3/wrist"]:
        prim = stage.GetPrimAtPath(joint_path)
        attr = prim.GetAttribute("mjc:actuatorgravcomp")
        assert attr.IsValid(), f"mjc:actuatorgravcomp not set on {joint_path}"
        assert attr.Get() is True


# -------------------------------------------------------------------
# End-to-end: IsaacLab config -> USD -> Newton -> MJCF XML
#
# This test builds a proper articulation on the USD stage, applies
# gravity compensation through the IsaacLab schema and actuator
# pipelines, then feeds the stage through Newton's ModelBuilder and
# MuJoCo solver, and finally verifies the exported MJCF XML.
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_end_to_end_isaaclab_to_mjcf(setup_simulation):
    """End-to-end: IsaacLab spawns articulation with gravity comp -> Newton -> MJCF XML has gravcomp."""
    stage = sim_utils.get_current_stage()

    # -- 1) Build articulation on the USD stage (simulating what a spawner does)
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")

    artic_prim = sim_utils.create_prim("/World/Robot", prim_type="Xform")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Robot"))

    # Body1: rigid body with collision
    body1 = sim_utils.create_prim("/World/Robot/Body1", prim_type="Xform")
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body1"))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body1"))
    col1 = sim_utils.create_prim("/World/Robot/Body1/Collision", prim_type="Cube")
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body1/Collision"))

    # Body2: rigid body with collision
    body2 = sim_utils.create_prim("/World/Robot/Body2", prim_type="Xform", translation=(1, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body2"))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body2"))
    col2 = sim_utils.create_prim("/World/Robot/Body2/Collision", prim_type="Sphere")
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body2/Collision"))

    # Joint1: connects world -> Body1
    joint1 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/Joint1")
    joint1.GetBody0Rel().SetTargets(["/World/Robot/Body1"])
    joint1.GetAxisAttr().Set("Z")
    joint1.GetLocalPos0Attr().Set((0, 0, 0))
    joint1.GetLocalPos1Attr().Set((0, 0, 0))
    joint1.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint1.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # Joint2: connects Body1 -> Body2
    joint2 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/Joint2")
    joint2.GetBody0Rel().SetTargets(["/World/Robot/Body1"])
    joint2.GetBody1Rel().SetTargets(["/World/Robot/Body2"])
    joint2.GetAxisAttr().Set("Y")
    joint2.GetLocalPos0Attr().Set((0, 0, 0))
    joint2.GetLocalPos1Attr().Set((0, 0, 0))
    joint2.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint2.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # -- 2) Apply gravity_compensation_scale via IsaacLab schemas (body-level)
    #    This is what the spawner does when rigid_props has gravity_compensation_scale
    rigid_cfg = schemas.RigidBodyPropertiesCfg(gravity_compensation_scale=0.5)
    schemas.modify_rigid_body_properties("/World/Robot", rigid_cfg)

    # -- 3) Apply actuator gravity_compensation via _write_actuator_gravity_comp_to_usd (joint-level)
    #    This is what the Newton Articulation class does before sim.reset()
    artic = object.__new__(Articulation)
    artic.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["Joint1"],
                gravity_compensation=True,
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
            "forearm": ImplicitActuatorCfg(
                joint_names_expr=["Joint2"],
                gravity_compensation=None,
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
        },
    )
    artic._write_actuator_gravity_comp_to_usd()

    # -- 4) Verify USD attributes were written by IsaacLab
    body1_prim = stage.GetPrimAtPath("/World/Robot/Body1")
    assert body1_prim.GetAttribute("mjc:gravcomp").IsValid(), "IsaacLab did not write mjc:gravcomp to Body1"

    joint1_prim = stage.GetPrimAtPath("/World/Robot/Joint1")
    assert joint1_prim.GetAttribute("mjc:actuatorgravcomp").IsValid(), (
        "IsaacLab did not write mjc:actuatorgravcomp to Joint1"
    )
    joint2_prim = stage.GetPrimAtPath("/World/Robot/Joint2")
    assert not joint2_prim.GetAttribute("mjc:actuatorgravcomp").IsValid(), (
        "mjc:actuatorgravcomp should not be on Joint2"
    )

    # -- 5) Feed the USD stage into Newton
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(stage)
    model = builder.finalize()

    # Verify Newton model parsed the attributes
    assert hasattr(model.mujoco, "gravcomp")
    gravcomp = model.mujoco.gravcomp.numpy()
    assert np.any(np.isclose(gravcomp, 0.5)), f"Newton model missing gravcomp=0.5, got {gravcomp}"

    assert hasattr(model.mujoco, "jnt_actgravcomp")
    jnt_actgravcomp = model.mujoco.jnt_actgravcomp.numpy()
    assert np.any(jnt_actgravcomp), f"Newton model missing jnt_actgravcomp=True, got {jnt_actgravcomp}"

    # -- 6) Build MuJoCo solver and export MJCF XML
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        mjcf_path = f.name
    try:
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=mjcf_path)
        root = ET.parse(mjcf_path).getroot()
    finally:
        os.unlink(mjcf_path)

    # NOTE: Newton's MJCF spec builder does not currently emit gravcomp on <body> elements
    # in the XML. Uncomment once Newton adds support for writing body gravcomp to the spec.
    # # Verify body-level gravcomp in MJCF XML
    # bodies = root.findall(".//body")
    # body_gravcomp_values = [float(b.get("gravcomp")) for b in bodies if b.get("gravcomp") is not None]
    # assert any(np.isclose(v, 0.5) for v in body_gravcomp_values), (
    #     f"MJCF XML missing gravcomp=0.5 on body, got {body_gravcomp_values}"
    # )

    # Verify joint-level actuatorgravcomp in MJCF XML
    joints = root.findall(".//joint")
    joint_actgravcomp_values = [j.get("actuatorgravcomp") for j in joints if j.get("actuatorgravcomp") is not None]
    assert any(v in ("true", "1") for v in joint_actgravcomp_values), (
        f"MJCF XML missing actuatorgravcomp=true on joint, got {joint_actgravcomp_values}"
    )
