# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext
from pxr import UsdPhysics

import isaaclab.sim.schemas as schemas
from isaaclab.sim.utils import find_global_fixed_joint_prim
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.string import to_camel_case


@pytest.fixture
def setup_simulation():
    """Fixture to set up and tear down the simulation context."""
    # Create a new stage
    stage_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="numpy")
    # Set some default values for test
    arti_cfg = schemas.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        articulation_enabled=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
        sleep_threshold=1.0,
        stabilization_threshold=5.0,
        fix_root_link=False,
    )
    rigid_cfg = schemas.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=False,
        disable_gravity=False,
        linear_damping=0.1,
        angular_damping=0.5,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=10.0,
        max_contact_impulse=10.0,
        enable_gyroscopic_forces=True,
        retain_accelerations=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
        sleep_threshold=1.0,
        stabilization_threshold=6.0,
    )
    collision_cfg = schemas.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.05,
        rest_offset=0.001,
        min_torsional_patch_radius=0.1,
        torsional_patch_radius=1.0,
    )
    mass_cfg = schemas.MassPropertiesCfg(mass=1.0, density=100.0)
    joint_cfg = schemas.JointDrivePropertiesCfg(
        drive_type="acceleration", max_effort=80.0, max_velocity=10.0, stiffness=10.0, damping=0.1
    )
    yield sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg
    # Teardown
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


def test_valid_properties_cfg(setup_simulation):
    """Test that all the config instances have non-None values.

    This is to ensure that we check that all the properties of the schema are set.
    """
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg = setup_simulation
    for cfg in [arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg]:
        # check nothing is none
        for k, v in cfg.__dict__.items():
            assert v is not None, f"{cfg.__class__.__name__}:{k} is None. Please make sure schemas are valid."


def test_modify_properties_on_invalid_prim(setup_simulation):
    """Test modifying properties on a prim that does not exist."""
    sim, _, rigid_cfg, _, _, _ = setup_simulation
    # set properties
    with pytest.raises(ValueError):
        schemas.modify_rigid_body_properties("/World/asset_xyz", rigid_cfg)


def test_modify_properties_on_articulation_instanced_usd(setup_simulation):
    """Test modifying properties on articulation instanced usd.

    In this case, modifying collision properties on the articulation instanced usd will fail.
    """
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg = setup_simulation
    # spawn asset to the stage
    asset_usd_file = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd"
    if "4.5" in ISAAC_NUCLEUS_DIR:
        asset_usd_file = asset_usd_file.replace("http", "https").replace("4.5", "5.0")
    prim_utils.create_prim("/World/asset_instanced", usd_path=asset_usd_file, translation=(0.0, 0.0, 0.62))

    # set properties on the asset and check all properties are set
    schemas.modify_articulation_root_properties("/World/asset_instanced", arti_cfg)
    schemas.modify_rigid_body_properties("/World/asset_instanced", rigid_cfg)
    schemas.modify_mass_properties("/World/asset_instanced", mass_cfg)
    schemas.modify_joint_drive_properties("/World/asset_instanced", joint_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/asset_instanced/base", arti_cfg, False)
    _validate_rigid_body_properties_on_prim("/World/asset_instanced", rigid_cfg)
    _validate_mass_properties_on_prim("/World/asset_instanced", mass_cfg)
    _validate_joint_drive_properties_on_prim("/World/asset_instanced", joint_cfg)

    # make a fixed joint
    arti_cfg.fix_root_link = True
    schemas.modify_articulation_root_properties("/World/asset_instanced", arti_cfg)


def test_modify_properties_on_articulation_usd(setup_simulation):
    """Test setting properties on articulation usd."""
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg = setup_simulation
    # spawn asset to the stage
    asset_usd_file = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    if "4.5" in ISAAC_NUCLEUS_DIR:
        asset_usd_file = asset_usd_file.replace("http", "https").replace("4.5", "5.0")
    prim_utils.create_prim("/World/asset", usd_path=asset_usd_file, translation=(0.0, 0.0, 0.62))

    # set properties on the asset and check all properties are set
    schemas.modify_articulation_root_properties("/World/asset", arti_cfg)
    schemas.modify_rigid_body_properties("/World/asset", rigid_cfg)
    schemas.modify_collision_properties("/World/asset", collision_cfg)
    schemas.modify_mass_properties("/World/asset", mass_cfg)
    schemas.modify_joint_drive_properties("/World/asset", joint_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/asset", arti_cfg, True)
    _validate_rigid_body_properties_on_prim("/World/asset", rigid_cfg)
    _validate_collision_properties_on_prim("/World/asset", collision_cfg)
    _validate_mass_properties_on_prim("/World/asset", mass_cfg)
    _validate_joint_drive_properties_on_prim("/World/asset", joint_cfg)

    # make a fixed joint
    arti_cfg.fix_root_link = True
    schemas.modify_articulation_root_properties("/World/asset", arti_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/asset", arti_cfg, True)


def test_defining_rigid_body_properties_on_prim(setup_simulation):
    """Test defining rigid body properties on a prim."""
    sim, _, rigid_cfg, collision_cfg, mass_cfg, _ = setup_simulation
    # create a prim
    prim_utils.create_prim("/World/parent", prim_type="XForm")
    # spawn a prim
    prim_utils.create_prim("/World/cube1", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    # set properties on the asset and check all properties are set
    schemas.define_rigid_body_properties("/World/cube1", rigid_cfg)
    schemas.define_collision_properties("/World/cube1", collision_cfg)
    schemas.define_mass_properties("/World/cube1", mass_cfg)
    # validate the properties
    _validate_rigid_body_properties_on_prim("/World/cube1", rigid_cfg)
    _validate_collision_properties_on_prim("/World/cube1", collision_cfg)
    _validate_mass_properties_on_prim("/World/cube1", mass_cfg)

    # spawn another prim
    prim_utils.create_prim("/World/cube2", prim_type="Cube", translation=(1.0, 1.0, 0.62))
    # set properties on the asset and check all properties are set
    schemas.define_rigid_body_properties("/World/cube2", rigid_cfg)
    schemas.define_collision_properties("/World/cube2", collision_cfg)
    # validate the properties
    _validate_rigid_body_properties_on_prim("/World/cube2", rigid_cfg)
    _validate_collision_properties_on_prim("/World/cube2", collision_cfg)

    # check if we can play
    sim.reset()
    for _ in range(100):
        sim.step()


def test_defining_articulation_properties_on_prim(setup_simulation):
    """Test defining articulation properties on a prim."""
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, _ = setup_simulation
    # create a parent articulation
    prim_utils.create_prim("/World/parent", prim_type="Xform")
    schemas.define_articulation_root_properties("/World/parent", arti_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/parent", arti_cfg, False)

    # create a child articulation
    prim_utils.create_prim("/World/parent/child", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_rigid_body_properties("/World/parent/child", rigid_cfg)
    schemas.define_mass_properties("/World/parent/child", mass_cfg)

    # check if we can play
    sim.reset()
    for _ in range(100):
        sim.step()


"""
Helper functions.
"""


def _validate_articulation_properties_on_prim(
    prim_path: str, arti_cfg, has_default_fixed_root: bool, verbose: bool = False
):
    """Validate the articulation properties on the prim.

    If :attr:`has_default_fixed_root` is True, then the asset already has a fixed root link. This is used to check the
    expected behavior of the fixed root link configuration.
    """
    # the root prim
    root_prim = prim_utils.get_prim_at_path(prim_path)
    # check articulation properties are set correctly
    for attr_name, attr_value in arti_cfg.__dict__.items():
        # skip names we know are not present
        if attr_name == "func":
            continue
        # handle fixed root link
        if attr_name == "fix_root_link" and attr_value is not None:
            # obtain the fixed joint prim
            fixed_joint_prim = find_global_fixed_joint_prim(prim_path)
            # if asset does not have a fixed root link then check if the joint is created
            if not has_default_fixed_root:
                if attr_value:
                    assert fixed_joint_prim is not None
                else:
                    assert fixed_joint_prim is None
            else:
                # check a joint exists
                assert fixed_joint_prim is not None
                # check if the joint is enabled or disabled
                is_enabled = fixed_joint_prim.GetJointEnabledAttr().Get()
                assert is_enabled == attr_value
            # skip the rest of the checks
            continue
        # convert attribute name in prim to cfg name
        prim_prop_name = f"physxArticulation:{to_camel_case(attr_name, to='cC')}"
        # validate the values
        assert root_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(
            attr_value, abs=1e-5
        ), f"Failed setting for {prim_prop_name}"


def _validate_rigid_body_properties_on_prim(prim_path: str, rigid_cfg, verbose: bool = False):
    """Validate the rigid body properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # the root prim
    root_prim = prim_utils.get_prim_at_path(prim_path)
    # check rigid body properties are set correctly
    for link_prim in root_prim.GetChildren():
        if UsdPhysics.RigidBodyAPI(link_prim):
            for attr_name, attr_value in rigid_cfg.__dict__.items():
                # skip names we know are not present
                if attr_name in ["func", "rigid_body_enabled", "kinematic_enabled"]:
                    continue
                # convert attribute name in prim to cfg name
                prim_prop_name = f"physxRigidBody:{to_camel_case(attr_name, to='cC')}"
                # validate the values
                assert link_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(
                    attr_value, abs=1e-5
                ), f"Failed setting for {prim_prop_name}"
        elif verbose:
            print(f"Skipping prim {link_prim.GetPrimPath()} as it is not a rigid body.")


def _validate_collision_properties_on_prim(prim_path: str, collision_cfg, verbose: bool = False):
    """Validate the collision properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # the root prim
    root_prim = prim_utils.get_prim_at_path(prim_path)
    # check collision properties are set correctly
    for link_prim in root_prim.GetChildren():
        for mesh_prim in link_prim.GetChildren():
            if UsdPhysics.CollisionAPI(mesh_prim):
                for attr_name, attr_value in collision_cfg.__dict__.items():
                    # skip names we know are not present
                    if attr_name in ["func", "collision_enabled"]:
                        continue
                    # convert attribute name in prim to cfg name
                    prim_prop_name = f"physxCollision:{to_camel_case(attr_name, to='cC')}"
                    # validate the values
                    assert mesh_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(
                        attr_value, abs=1e-5
                    ), f"Failed setting for {prim_prop_name}"
            elif verbose:
                print(f"Skipping prim {mesh_prim.GetPrimPath()} as it is not a collision mesh.")


def _validate_mass_properties_on_prim(prim_path: str, mass_cfg, verbose: bool = False):
    """Validate the mass properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # the root prim
    root_prim = prim_utils.get_prim_at_path(prim_path)
    # check rigid body mass properties are set correctly
    for link_prim in root_prim.GetChildren():
        if UsdPhysics.MassAPI(link_prim):
            for attr_name, attr_value in mass_cfg.__dict__.items():
                # skip names we know are not present
                if attr_name in ["func"]:
                    continue
                # print(link_prim.GetProperties())
                prim_prop_name = f"physics:{to_camel_case(attr_name, to='cC')}"
                # validate the values
                assert link_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(
                    attr_value, abs=1e-5
                ), f"Failed setting for {prim_prop_name}"
        elif verbose:
            print(f"Skipping prim {link_prim.GetPrimPath()} as it is not a mass api.")


def _validate_joint_drive_properties_on_prim(prim_path: str, joint_cfg, verbose: bool = False):
    """Validate the mass properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # the root prim
    root_prim = prim_utils.get_prim_at_path(prim_path)
    # check joint drive properties are set correctly
    for link_prim in root_prim.GetAllChildren():
        for joint_prim in link_prim.GetChildren():
            if joint_prim.IsA(UsdPhysics.PrismaticJoint) or joint_prim.IsA(UsdPhysics.RevoluteJoint):
                # check it has drive API
                assert joint_prim.HasAPI(UsdPhysics.DriveAPI)
                # iterate over the joint properties
                for attr_name, attr_value in joint_cfg.__dict__.items():
                    # skip names we know are not present
                    if attr_name == "func":
                        continue
                    # resolve the drive (linear or angular)
                    drive_model = "linear" if joint_prim.IsA(UsdPhysics.PrismaticJoint) else "angular"

                    # manually check joint type since it is a string type
                    if attr_name == "drive_type":
                        prim_attr_name = f"drive:{drive_model}:physics:type"
                        # check the value
                        assert attr_value == joint_prim.GetAttribute(prim_attr_name).Get()
                        continue

                    # non-string attributes
                    if attr_name == "max_velocity":
                        prim_attr_name = "physxJoint:maxJointVelocity"
                    elif attr_name == "max_effort":
                        prim_attr_name = f"drive:{drive_model}:physics:maxForce"
                    else:
                        prim_attr_name = f"drive:{drive_model}:physics:{to_camel_case(attr_name, to='cC')}"

                    # obtain value from USD API (for angular, these follow degrees unit)
                    prim_attr_value = joint_prim.GetAttribute(prim_attr_name).Get()

                    # for angular drives, we expect user to set in radians
                    # the values reported by USD are in degrees
                    if drive_model == "angular":
                        if attr_name == "max_velocity":
                            # deg / s --> rad / s
                            prim_attr_value = prim_attr_value * math.pi / 180.0
                        elif attr_name in ["stiffness", "damping"]:
                            # N-m/deg or N-m-s/deg --> N-m/rad or N-m-s/rad
                            prim_attr_value = prim_attr_value * 180.0 / math.pi

                    # validate the values
                    assert prim_attr_value == pytest.approx(
                        attr_value, abs=1e-5
                    ), f"Failed setting for {prim_attr_name}"
            elif verbose:
                print(f"Skipping prim {joint_prim.GetPrimPath()} as it is not a joint drive api.")
