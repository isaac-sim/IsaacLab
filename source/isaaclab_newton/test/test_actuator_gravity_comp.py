# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for gravity compensation USD attributes and their propagation to MJCF."""

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
from newton.solvers import SolverMuJoCo

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
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


# -------------------------------------------------------------------
# Tests: Joint-level gravity compensation via schemas
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_gravity_comp_written_to_joint(setup_simulation):
    """Test that mjc:actuatorgravcomp is set on joint prims via modify_joint_drive_properties."""
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Robot", prim_type="Xform")
    sim_utils.create_prim("/World/Robot/body0", prim_type="Cube")
    sim_utils.create_prim("/World/Robot/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/body1/joint0")

    joint_cfg = schemas.JointDrivePropertiesCfg(gravity_compensation=True)
    schemas.modify_joint_drive_properties("/World/Robot", joint_cfg)

    prim = stage.GetPrimAtPath("/World/Robot/body1/joint0")
    attr = prim.GetAttribute("mjc:actuatorgravcomp")
    assert attr.IsValid(), "mjc:actuatorgravcomp not set on joint"
    assert attr.Get() is True


@pytest.mark.isaacsim_ci
def test_gravity_comp_not_written_when_none(setup_simulation):
    """Test that no attributes are written when gravity_compensation is None."""
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Robot", prim_type="Xform")
    sim_utils.create_prim("/World/Robot/body0", prim_type="Cube")
    sim_utils.create_prim("/World/Robot/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/body1/joint0")

    joint_cfg = schemas.JointDrivePropertiesCfg()
    schemas.modify_joint_drive_properties("/World/Robot", joint_cfg)

    prim = stage.GetPrimAtPath("/World/Robot/body1/joint0")
    attr = prim.GetAttribute("mjc:actuatorgravcomp")
    assert not attr.IsValid(), "mjc:actuatorgravcomp should not be set when gravity_compensation is None"


# -------------------------------------------------------------------
# End-to-end: IsaacLab config -> USD -> Newton -> MJCF XML
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_end_to_end_isaaclab_to_mjcf(setup_simulation):
    """End-to-end: IsaacLab spawns articulation with gravity comp -> Newton -> MJCF XML has gravcomp."""
    stage = sim_utils.get_current_stage()

    # -- 1) Build articulation on the USD stage
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")

    sim_utils.create_prim("/World/Robot", prim_type="Xform")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Robot"))

    # Body1: rigid body with collision
    sim_utils.create_prim("/World/Robot/Body1", prim_type="Xform")
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body1"))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body1"))
    sim_utils.create_prim("/World/Robot/Body1/Collision", prim_type="Cube")
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body1/Collision"))

    # Body2: rigid body with collision
    sim_utils.create_prim("/World/Robot/Body2", prim_type="Xform", translation=(1, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body2"))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/Robot/Body2"))
    sim_utils.create_prim("/World/Robot/Body2/Collision", prim_type="Sphere")
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
    rigid_cfg = schemas.RigidBodyPropertiesCfg(gravity_compensation_scale=0.5)
    schemas.modify_rigid_body_properties("/World/Robot", rigid_cfg)

    # -- 3) Apply joint-level gravity compensation via schemas
    joint_cfg = schemas.JointDrivePropertiesCfg(gravity_compensation=True)
    schemas.modify_joint_drive_properties("/World/Robot/Joint1", joint_cfg)

    # -- 4) Verify USD attributes were written
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
        SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=mjcf_path)
        root = ET.parse(mjcf_path).getroot()
    finally:
        os.unlink(mjcf_path)

    # Verify body-level gravcomp in MJCF XML
    bodies = root.findall(".//body")
    body_gravcomp_values = [float(b.get("gravcomp")) for b in bodies if b.get("gravcomp") is not None]
    assert any(np.isclose(v, 0.5) for v in body_gravcomp_values), (
        f"MJCF XML missing gravcomp=0.5 on body, got {body_gravcomp_values}"
    )

    # Verify joint-level actuatorgravcomp in MJCF XML
    joints = root.findall(".//joint")
    joint_actgravcomp_values = [j.get("actuatorgravcomp") for j in joints if j.get("actuatorgravcomp") is not None]
    assert any(v in ("true", "1") for v in joint_actgravcomp_values), (
        f"MJCF XML missing actuatorgravcomp=true on joint, got {joint_actgravcomp_values}"
    )
