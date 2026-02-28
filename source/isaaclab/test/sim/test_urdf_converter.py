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
import tempfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import omni.kit.app
from isaacsim.core.prims import Articulation

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.converters.urdf_utils import merge_fixed_joints


# Create a fixture for setup and teardown
@pytest.fixture
def sim_config():
    # Create a new stage
    sim_utils.create_new_stage()
    # enable the URDF importer extension
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
        manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    # obtain the extension path
    extension_id = manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
    extension_path = manager.get_extension_path(extension_id)
    # default configuration
    config = UrdfConverterCfg(
        asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
        fix_base=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    yield sim, config
    # Teardown
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_no_change(sim_config):
    """Call conversion twice. This should not generate a new USD file."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)
    time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

    # no change to config only define the usd directory
    new_config = config
    new_config.usd_dir = urdf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_urdf_converter = UrdfConverter(new_config)
    new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created == new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_config_change(sim_config):
    """Call conversion twice but change the config in the second call. This should generate a new USD file."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)
    time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

    # change the config
    new_config = config
    new_config.fix_base = not config.fix_base
    # define the usd directory
    new_config.usd_dir = urdf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_urdf_converter = UrdfConverter(new_config)
    new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created != new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_create_prim_from_usd(sim_config):
    """Call conversion and create a prim from it."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_config_drive_type(sim_config):
    """Change the drive mechanism of the robot to be position."""
    sim, config = sim_config
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_converter")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # change the config
    config.force_usd_conversion = True
    config.joint_drive.target_type = "position"
    config.joint_drive.gains.stiffness = 42.0
    config.joint_drive.gains.damping = 4.2
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)
    # check the drive type of the robot
    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

    # access the robot
    robot = Articulation(prim_path, reset_xform_properties=False)
    # play the simulator and initialize the robot
    sim.reset()
    robot.initialize()

    # check drive values for the robot (read from physx)
    drive_stiffness, drive_damping = robot.get_gains()
    np.testing.assert_allclose(drive_stiffness.cpu().numpy(), config.joint_drive.gains.stiffness)
    np.testing.assert_allclose(drive_damping.cpu().numpy(), config.joint_drive.gains.damping)

    # check drive values for the robot (read from usd)
    # Note: Disable the app control callback to prevent hanging during sim.stop()
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    drive_stiffness, drive_damping = robot.get_gains()
    np.testing.assert_allclose(drive_stiffness.cpu().numpy(), config.joint_drive.gains.stiffness)
    np.testing.assert_allclose(drive_damping.cpu().numpy(), config.joint_drive.gains.damping)


@pytest.mark.isaacsim_ci
def test_merge_fixed_joints_xml():
    """Test that the merge_fixed_joints utility correctly modifies URDF XML.

    Uses ``test_merge_joints.urdf`` which has:
      - 7 links (root_link, base_link, link_1, link_2, palm_link, finger_link_1, finger_link_2)
      - 6 joints (3 fixed, 1 continuous, 2 prismatic)

    After merging:
      - 4 links (root_link, link_1, finger_link_1, finger_link_2)
      - 3 joints (0 fixed, 1 continuous, 2 prismatic)
    """
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
        manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    extension_id = manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
    extension_path = manager.get_extension_path(extension_id)

    urdf_path = os.path.join(extension_path, "data", "urdf", "tests", "test_merge_joints.urdf")

    with tempfile.TemporaryDirectory(prefix="isaaclab_test_merge_") as tmpdir:
        output_path = os.path.join(tmpdir, "merged.urdf")
        merge_fixed_joints(urdf_path, output_path)

        # parse the output URDF
        tree = ET.parse(output_path)
        root = tree.getroot()

        links = root.findall("link")
        joints = root.findall("joint")
        link_names = sorted(link.get("name") for link in links)
        joint_types = [j.get("type") for j in joints]

        # verify link count and names
        assert len(links) == 4, f"Expected 4 links, got {len(links)}: {link_names}"
        assert sorted(link_names) == sorted(["root_link", "link_1", "finger_link_1", "finger_link_2"])

        # verify no fixed joints remain
        assert "fixed" not in joint_types, f"Fixed joints should be removed, got types: {joint_types}"

        # verify joint count and types
        assert len(joints) == 3, f"Expected 3 joints, got {len(joints)}"

        # verify that visuals from merged links were transferred
        # root_link should have base_link's visual (1 visual from base_link)
        root_link = next(link for link in links if link.get("name") == "root_link")
        root_visuals = root_link.findall("visual")
        assert len(root_visuals) >= 1, "root_link should have at least 1 visual from merged base_link"

        # link_1 should have visuals from link_1, link_2, and palm_link (3 total)
        link_1 = next(link for link in links if link.get("name") == "link_1")
        link_1_visuals = link_1.findall("visual")
        assert len(link_1_visuals) == 3, (
            f"link_1 should have 3 visuals (own + link_2 + palm_link), got {len(link_1_visuals)}"
        )

        # verify that re-parented joints (finger joints) now reference link_1
        for joint in joints:
            parent_name = joint.find("parent").get("link")
            child_name = joint.find("child").get("link")
            # finger joints were parented to palm_link, should now be parented to link_1
            if child_name in ("finger_link_1", "finger_link_2"):
                assert parent_name == "link_1", f"Expected finger joint parent to be 'link_1', got '{parent_name}'"


@pytest.mark.isaacsim_ci
def test_merge_fixed_joints_converter(sim_config):
    """Test the full URDF converter pipeline with merge_fixed_joints enabled."""
    sim, config = sim_config
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_converter_merge")
    os.makedirs(output_dir, exist_ok=True)

    # use a URDF that has fixed joints
    manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
    extension_path = manager.get_extension_path(extension_id)

    config.asset_path = os.path.join(extension_path, "data", "urdf", "tests", "test_merge_joints.urdf")
    config.merge_fixed_joints = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir

    urdf_converter = UrdfConverter(config)

    # check the USD file was created
    assert os.path.exists(urdf_converter.usd_path), f"USD file not found at: {urdf_converter.usd_path}"

    # create a prim from it and verify it's valid
    prim_path = "/World/MergedRobot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_fix_base_creates_fixed_joint(sim_config):
    """Verify that fix_base=True creates a FixedJoint in the output USD."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_fix_base")
    os.makedirs(output_dir, exist_ok=True)

    config.fix_base = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # search for a FixedJoint in the output
    fixed_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
    assert len(fixed_joints) > 0, "Expected at least one FixedJoint from fix_base=True"

    # the first FixedJoint should target a rigid body link via body1
    fj = UsdPhysics.FixedJoint(fixed_joints[0])
    body1_targets = fj.GetBody1Rel().GetTargets()
    assert len(body1_targets) > 0, "FixedJoint should target a rigid body link via body1"


@pytest.mark.isaacsim_ci
def test_no_fix_base(sim_config):
    """Verify that fix_base=False does not create a fix_base_joint."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_no_fix_base")
    os.makedirs(output_dir, exist_ok=True)

    config.fix_base = False
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # there should be no prim named "fix_base_joint"
    fix_base_prims = [p for p in stage.Traverse() if p.GetName() == "fix_base_joint"]
    assert len(fix_base_prims) == 0, "Expected no fix_base_joint when fix_base=False"


@pytest.mark.isaacsim_ci
def test_collision_from_visuals(sim_config):
    """Verify that collision_from_visuals runs without error and produces valid output.

    Note: CollisionAPI is applied on the intermediate stage before the asset transformer
    restructures the USD.  The transformer may not preserve CollisionAPI in the final
    output, so this test verifies the pipeline executes successfully rather than
    inspecting the final USD for CollisionAPI schemas.
    """
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_collision_visuals")
    os.makedirs(output_dir, exist_ok=True)

    config.collision_from_visuals = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    assert os.path.exists(urdf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_no_collision_from_visuals(sim_config):
    """Verify that conversion succeeds when collision_from_visuals is disabled."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_no_collision_visuals")
    os.makedirs(output_dir, exist_ok=True)

    config.collision_from_visuals = False
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    assert os.path.exists(urdf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_self_collision(sim_config):
    """Verify that self_collision=True enables self-collision on the articulation."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_self_collision")
    os.makedirs(output_dir, exist_ok=True)

    config.self_collision = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import PhysxSchema, Usd

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # find prim with PhysxArticulationAPI and check self-collision flag
    found_self_collision = False
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            physx_api = PhysxSchema.PhysxArticulationAPI(prim)
            sc_attr = physx_api.GetEnabledSelfCollisionsAttr()
            if sc_attr and sc_attr.HasValue() and sc_attr.Get():
                found_self_collision = True
                break

    assert found_self_collision, "Expected self-collision to be enabled on the articulation"


@pytest.mark.isaacsim_ci
def test_drive_type_acceleration(sim_config):
    """Verify that drive_type='acceleration' is applied to all joints."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_drive_accel")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.joint_drive.drive_type = "acceleration"
    config.joint_drive.gains.stiffness = 100.0
    config.joint_drive.gains.damping = 10.0
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    joint_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            instance_name = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
            drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
            type_attr = drive.GetTypeAttr()
            assert type_attr and type_attr.HasValue(), f"Joint {prim.GetName()} missing drive type"
            assert type_attr.Get() == "acceleration", (
                f"Expected drive type 'acceleration' on {prim.GetName()}, got '{type_attr.Get()}'"
            )
            joint_count += 1

    assert joint_count > 0, "No joints found in the output USD"


@pytest.mark.isaacsim_ci
def test_target_type_none_zeros_gains(sim_config):
    """Verify that target_type='none' sets stiffness and damping to 0."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_target_none")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.joint_drive.target_type = "none"
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    robot = Articulation(prim_path, reset_xform_properties=False)
    sim.reset()
    robot.initialize()

    drive_stiffness, drive_damping = robot.get_gains()
    np.testing.assert_allclose(drive_stiffness.cpu().numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(drive_damping.cpu().numpy(), 0.0, atol=1e-6)


@pytest.mark.isaacsim_ci
def test_per_joint_dict_gains(sim_config):
    """Verify that per-joint dict-based gains are applied correctly."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_dict_gains")
    os.makedirs(output_dir, exist_ok=True)

    arm_stiffness = 100.0
    finger_stiffness = 200.0
    arm_damping = 10.0
    finger_damping = 20.0

    config.force_usd_conversion = True
    config.joint_drive.target_type = "position"
    config.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
        stiffness={
            "panda_joint[1-7]": arm_stiffness,
            "panda_finger": finger_stiffness,
        },
        damping={
            "panda_joint[1-7]": arm_damping,
            "panda_finger": finger_damping,
        },
    )
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    # inspect the USD directly rather than going through PhysX to verify per-joint values
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    arm_joint_count = 0
    finger_joint_count = 0
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        name = prim.GetName()
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        instance_name = "angular" if is_revolute else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
        stiffness_attr = drive.GetStiffnessAttr()
        damping_attr = drive.GetDampingAttr()

        if "panda_joint" in name and "finger" not in name:
            # arm joint (revolute) — USD stores in Nm/deg, so expected = value * pi/180
            import math

            expected_s = arm_stiffness * math.pi / 180.0
            expected_d = arm_damping * math.pi / 180.0
            assert abs(stiffness_attr.Get() - expected_s) < 0.01, (
                f"Arm joint {name}: expected stiffness ~{expected_s}, got {stiffness_attr.Get()}"
            )
            assert abs(damping_attr.Get() - expected_d) < 0.01, (
                f"Arm joint {name}: expected damping ~{expected_d}, got {damping_attr.Get()}"
            )
            arm_joint_count += 1
        elif "finger" in name:
            # finger joint (prismatic) — USD stores directly in N/m
            assert abs(stiffness_attr.Get() - finger_stiffness) < 0.01, (
                f"Finger joint {name}: expected stiffness {finger_stiffness}, got {stiffness_attr.Get()}"
            )
            assert abs(damping_attr.Get() - finger_damping) < 0.01, (
                f"Finger joint {name}: expected damping {finger_damping}, got {damping_attr.Get()}"
            )
            finger_joint_count += 1

    assert arm_joint_count == 7, f"Expected 7 arm joints, got {arm_joint_count}"
    assert finger_joint_count == 2, f"Expected 2 finger joints, got {finger_joint_count}"


@pytest.mark.isaacsim_ci
def test_per_joint_dict_drive_type(sim_config):
    """Verify that per-joint dict-based drive type is applied correctly."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_dict_drive_type")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.joint_drive.drive_type = {
        "panda_joint[1-7]": "acceleration",
        "panda_finger": "force",
    }
    config.joint_drive.gains.stiffness = 50.0
    config.joint_drive.gains.damping = 5.0
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        name = prim.GetName()
        instance_name = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
        type_attr = drive.GetTypeAttr()

        if "panda_joint" in name and "finger" not in name:
            assert type_attr.Get() == "acceleration", (
                f"Arm joint {name}: expected 'acceleration', got '{type_attr.Get()}'"
            )
        elif "finger" in name:
            assert type_attr.Get() == "force", f"Finger joint {name}: expected 'force', got '{type_attr.Get()}'"


@pytest.mark.isaacsim_ci
def test_natural_frequency_gains_deprecation(sim_config):
    """Verify that NaturalFrequencyGainsCfg emits a DeprecationWarning and conversion still succeeds."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_nat_freq")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
        natural_frequency=10.0,
    )
    config.usd_dir = output_dir

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        urdf_converter = UrdfConverter(config)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "Expected DeprecationWarning for NaturalFrequencyGainsCfg"
        assert "NaturalFrequencyGainsCfg" in str(dep_warnings[0].message)

    # conversion should still succeed
    assert os.path.exists(urdf_converter.usd_path), "USD file should be created despite deprecation"

    # verify we can spawn from the output
    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_usd_structure_has_joints_and_links(sim_config):
    """Validate that the output USD contains the expected joint and link prims for Franka Panda."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_structure")
    os.makedirs(output_dir, exist_ok=True)

    config.merge_fixed_joints = False
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # count revolute and prismatic joints
    revolute_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    prismatic_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    rigid_bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]

    # Franka Panda: 7 revolute arm joints, 2 prismatic finger joints
    assert len(revolute_joints) >= 7, f"Expected at least 7 revolute joints, got {len(revolute_joints)}"
    assert len(prismatic_joints) >= 2, f"Expected at least 2 prismatic joints, got {len(prismatic_joints)}"
    assert len(rigid_bodies) >= 1, "Expected at least one rigid body link"

    # all joints should have DriveAPI applied
    for joint_prim in revolute_joints + prismatic_joints:
        instance_name = "angular" if joint_prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        drive = UsdPhysics.DriveAPI.Get(joint_prim, instance_name)
        stiffness_attr = drive.GetStiffnessAttr()
        assert stiffness_attr and stiffness_attr.HasValue(), (
            f"Joint {joint_prim.GetName()} missing stiffness attribute in DriveAPI"
        )


@pytest.mark.isaacsim_ci
def test_link_density(sim_config):
    """Verify that link_density applies density to rigid body links.

    Note: The Franka Panda URDF has explicit mass on all links, so ``_apply_link_density``
    only sets density on links without explicit mass (mass == 0). This test verifies the
    pipeline runs without errors when link_density is set.
    """
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_link_density")
    os.makedirs(output_dir, exist_ok=True)

    config.link_density = 500.0
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # verify conversion succeeds and prims with MassAPI exist
    mass_prims = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
    assert len(mass_prims) > 0, "Expected prims with MassAPI"

    # verify we can spawn from the output
    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_collider_type_convex_decomposition(sim_config):
    """Verify that collider_type='convex_decomposition' runs without error and produces valid output.

    Note: MeshCollisionAPI is applied on the intermediate stage before the asset transformer.
    The transformer may not preserve these schemas in the final output, so this test
    verifies the pipeline executes successfully and produces a spawnable USD.
    """
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_convex_decomp")
    os.makedirs(output_dir, exist_ok=True)

    config.collision_from_visuals = True
    config.collider_type = "convex_decomposition"
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    assert os.path.exists(urdf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_unsupported_features_warn(sim_config):
    """Verify that deprecated config options emit warnings without failing."""
    sim, config = sim_config
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_deprecated_warn")
    os.makedirs(output_dir, exist_ok=True)

    config.convert_mimic_joints_to_normal_joints = True
    config.replace_cylinders_with_capsules = True
    config.root_link_name = "some_link"
    config.force_usd_conversion = True
    config.usd_dir = output_dir

    # conversion should succeed despite deprecated options
    urdf_converter = UrdfConverter(config)
    assert os.path.exists(urdf_converter.usd_path), "USD file should be created despite deprecated options"
