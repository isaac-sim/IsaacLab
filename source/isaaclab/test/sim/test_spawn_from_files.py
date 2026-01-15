# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

"""Launch Isaac Sim Simulator first."""

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
from packaging.version import Version

import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.version import get_isaac_sim_version


@pytest.fixture
def sim():
    """Create a blank new stage for each test."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    # Wait for spawning
    sim_utils.update_stage()

    yield sim

    # cleanup after test
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_spawn_usd(sim):
    """Test loading prim from Usd file."""
    # Spawn cone
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim = cfg.func("/World/Franka", cfg)
    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Franka").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"


@pytest.mark.isaacsim_ci
def test_spawn_usd_fails(sim):
    """Test loading prim from Usd file fails when asset usd path is invalid."""
    # Spawn cone
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda2_instanceable.usd")

    with pytest.raises(FileNotFoundError):
        cfg.func("/World/Franka", cfg)


@pytest.mark.isaacsim_ci
def test_spawn_urdf(sim):
    """Test loading prim from URDF file."""
    # pin the urdf importer extension to the older version
    manager = omni.kit.app.get_app().get_extension_manager()
    if get_isaac_sim_version() >= Version("5.1"):
        pinned_urdf_extension_name = "isaacsim.asset.importer.urdf-2.4.31"
        manager.set_extension_enabled_immediate(pinned_urdf_extension_name, True)
    else:
        pinned_urdf_extension_name = "isaacsim.asset.importer.urdf"
    # retrieve path to urdf importer extension
    extension_id = manager.get_enabled_extension_id(pinned_urdf_extension_name)
    extension_path = manager.get_extension_path(extension_id)
    # Spawn franka from URDF
    cfg = sim_utils.UrdfFileCfg(
        asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    prim = cfg.func("/World/Franka", cfg)
    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Franka").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"


@pytest.mark.isaacsim_ci
def test_spawn_ground_plane(sim):
    """Test loading prim for the ground plane from grid world USD."""
    # Spawn ground plane
    cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(10.0, 10.0))
    prim = cfg.func("/World/ground_plane", cfg)
    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/ground_plane").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"


@pytest.mark.isaacsim_ci
def test_spawn_usd_with_compliant_contact_material(sim):
    """Test loading prim from USD file with physics material applied to specific prim."""
    # Spawn gelsight finger with physics material on specific prim
    usd_file_path = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"

    # Create spawn configuration
    spawn_cfg = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        compliant_contact_stiffness=1000.0,
        compliant_contact_damping=100.0,
        physics_material_prim_path="elastomer",
    )

    # Spawn the prim
    prim = spawn_cfg.func("/World/Robot", spawn_cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Robot").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"

    material_prim_path = "/World/Robot/elastomer/compliant_material"
    # Check that the physics material was applied to the specified prim
    assert sim.stage.GetPrimAtPath(material_prim_path).IsValid()

    # Check properties
    material_prim = sim.stage.GetPrimAtPath(material_prim_path)
    assert material_prim.IsValid()
    assert material_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == 1000.0
    assert material_prim.GetAttribute("physxMaterial:compliantContactDamping").Get() == 100.0


@pytest.mark.isaacsim_ci
def test_spawn_usd_with_compliant_contact_material_on_multiple_prims(sim):
    """Test loading prim from USD file with physics material applied to multiple prims."""
    # Spawn Panda robot with physics material on specific prims
    usd_file_path = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"

    # Create spawn configuration
    spawn_cfg = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        compliant_contact_stiffness=1000.0,
        compliant_contact_damping=100.0,
        physics_material_prim_path=["elastomer", "gelsight_finger"],
    )

    # Spawn the prim
    prim = spawn_cfg.func("/World/Robot", spawn_cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Robot").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"

    # Check that the physics material was applied to the specified prims
    for link_name in ["elastomer", "gelsight_finger"]:
        material_prim_path = f"/World/Robot/{link_name}/compliant_material"
        print("checking", material_prim_path)
        assert sim.stage.GetPrimAtPath(material_prim_path).IsValid()

        # Check properties
        material_prim = sim.stage.GetPrimAtPath(material_prim_path)
        assert material_prim.IsValid()
        assert material_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == 1000.0
        assert material_prim.GetAttribute("physxMaterial:compliantContactDamping").Get() == 100.0


@pytest.mark.isaacsim_ci
def test_spawn_usd_with_compliant_contact_material_no_prim_path(sim):
    """Test loading prim from USD file with physics material but no prim path specified."""
    # Spawn gelsight finger without specifying prim path for physics material
    usd_file_path = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"

    # Create spawn configuration without physics material prim path
    spawn_cfg = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        compliant_contact_stiffness=1000.0,
        compliant_contact_damping=100.0,
        physics_material_prim_path=None,
    )

    # Spawn the prim
    prim = spawn_cfg.func("/World/Robot", spawn_cfg)

    # Check validity - should still spawn successfully but without physics material
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Robot").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"

    material_prim_path = "/World/Robot/elastomer/compliant_material"
    material_prim = sim.stage.GetPrimAtPath(material_prim_path)
    assert material_prim is not None
    assert not material_prim.IsValid()
