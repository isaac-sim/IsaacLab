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

import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


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
    # enable the URDF importer extension
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
        manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    # retrieve path to urdf importer extension
    extension_id = manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
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


@pytest.mark.isaacsim_ci
def test_spawn_usd_collision_props_applied_to_instanced_prims(sim):
    """Test that collision_props are applied to ALL collision meshes, including instanced ones.

    This is a regression test for an issue where @apply_nested skipped instanced prims,
    causing collision properties to not be applied to robot collision meshes that were
    USD instances.
    """
    from pxr import Usd, UsdPhysics

    # Spawn instanceable robot with collision_props
    # The panda_instanceable.usd has instanced collision meshes
    rest_offset_value = 0.05
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        collision_props=sim_utils.CollisionPropertiesCfg(rest_offset=rest_offset_value),
    )
    prim = cfg.func("/World/Franka", cfg)

    # Check validity
    assert prim.IsValid()

    # Find all collision meshes under the robot and verify rest_offset is applied
    collision_mesh_count = 0
    props_applied_count = 0

    # Use Usd.PrimRange with TraverseInstanceProxies to traverse into instanced prims
    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        # Check if this prim has collision API
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            collision_mesh_count += 1
            # Check if rest_offset was applied (use approximate comparison for float32 precision)
            rest_offset_attr = descendant.GetAttribute("physxCollision:restOffset")
            if rest_offset_attr:
                actual_value = rest_offset_attr.Get()
                if actual_value is not None and abs(actual_value - rest_offset_value) < 1e-6:
                    props_applied_count += 1

    # There should be collision meshes in the robot
    assert collision_mesh_count > 0, "Robot should have collision meshes"

    # ALL collision meshes should have the rest_offset applied
    assert props_applied_count == collision_mesh_count, (
        f"collision_props not applied to all collision meshes: "
        f"{props_applied_count}/{collision_mesh_count} have rest_offset={rest_offset_value}. "
        "This may indicate instanced prims are being skipped."
    )


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "material_cfg,material_name",
    [
        (sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), "PreviewSurface"),
        (sim_utils.GlassMdlCfg(glass_color=(0.0, 1.0, 0.0)), "GlassMdl"),
    ],
)
def test_spawn_usd_visual_material_binding_on_instanced_prims(sim, material_cfg, material_name):
    """Test that visual_material binding propagates to ALL meshes, including instanced ones.

    This tests whether USD material binding inheritance works through instance proxies
    for different material types (PreviewSurface and MDL materials).
    """
    from pxr import Usd, UsdGeom, UsdShade

    # Spawn instanceable robot with visual_material
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        visual_material=material_cfg,
        visual_material_path=f"Looks/Test{material_name}",
    )
    prim = cfg.func("/World/Franka", cfg)

    # Check validity
    assert prim.IsValid()

    # Get the material path
    material_path = f"/World/Franka/Looks/Test{material_name}"
    stage = prim.GetStage()
    material_prim = stage.GetPrimAtPath(material_path)
    assert material_prim.IsValid(), f"Material prim should exist at {material_path}"

    # Find all mesh prims under the robot and check material binding
    mesh_count = 0
    correct_binding_count = 0
    wrong_binding_meshes = []

    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if descendant.IsA(UsdGeom.Mesh):
            mesh_count += 1
            # Check material binding - verify it's OUR material, not pre-existing
            binding_api = UsdShade.MaterialBindingAPI(descendant)
            bound_material, _ = binding_api.ComputeBoundMaterial()
            if bound_material:
                bound_path = bound_material.GetPrim().GetPath()
                if str(bound_path) == material_path:
                    correct_binding_count += 1
                else:
                    wrong_binding_meshes.append((str(descendant.GetPath()), str(bound_path)))
            else:
                wrong_binding_meshes.append((str(descendant.GetPath()), "None"))

    # There should be meshes in the robot
    assert mesh_count > 0, "Robot should have mesh prims"

    # Log results for debugging
    print(f"{material_name} binding test: {correct_binding_count}/{mesh_count} meshes bound to our material")
    if wrong_binding_meshes:
        print(f"Wrong/missing bindings (first 5): {wrong_binding_meshes[:5]}")

    # ALL meshes should be bound to OUR material (not pre-existing ones)
    # This verifies that bind_visual_material with stronger_than_descendants=True
    # actually overrides existing bindings on all descendants including instance proxies
    assert correct_binding_count == mesh_count, (
        f"{material_name}: Only {correct_binding_count}/{mesh_count} meshes bound to our material. "
        f"Expected all meshes to use {material_path}. "
        f"Wrong bindings: {wrong_binding_meshes[:5]}"
    )
