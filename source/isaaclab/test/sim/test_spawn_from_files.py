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
import trimesh

import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas import MESH_APPROXIMATION_TOKENS
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


_TRIANGLE_VERTICES = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
_TRIANGLE_FACES = [(0, 1, 2)]


def _triangle_mesh_cfg(**kwargs):
    return sim_utils.MeshFileCfg(
        mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=_TRIANGLE_VERTICES, faces=_TRIANGLE_FACES),
        **kwargs,
    )


@pytest.mark.isaacsim_ci
def test_spawn_mesh_file_with_trimesh_object(sim):
    """Test spawning an in-memory mesh with object-style collision and rigid properties."""
    cfg = sim_utils.MeshFileCfg(
        mesh=sim_utils.MeshFileCfg.TrimeshObjectCfg(mesh=trimesh.creation.box()),
        scale=(1.5, 1.0, 0.5),
        mesh_collision_props=sim_utils.ConvexHullPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
    )
    prim = cfg.func("/World/Object", cfg)

    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Object").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"

    root_prim = sim.stage.GetPrimAtPath("/World/Object")
    assert root_prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert root_prim.GetAttribute("physics:mass").Get() == pytest.approx(2.0)

    mesh_prim = sim.stage.GetPrimAtPath("/World/Object/mesh")
    assert mesh_prim.IsValid()
    assert UsdPhysics.CollisionAPI(mesh_prim).GetCollisionEnabledAttr().Get() is True
    assert (
        UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get() == MESH_APPROXIMATION_TOKENS["convexHull"]
    )


@pytest.mark.isaacsim_ci
def test_spawn_mesh_file_with_triangle_mesh_data(sim):
    """Test spawning generated triangle mesh data as a visual-only mesh."""
    cfg = sim_utils.MeshFileCfg(
        mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(
            vertices=_TRIANGLE_VERTICES,
            faces=_TRIANGLE_FACES,
            vertex_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        )
    )
    prim = cfg.func("/World/VisualMesh", cfg)

    assert prim.IsValid()
    mesh_prim = sim.stage.GetPrimAtPath("/World/VisualMesh/mesh")
    assert mesh_prim.IsValid()
    assert not UsdPhysics.CollisionAPI(mesh_prim)
    assert not UsdPhysics.MeshCollisionAPI(mesh_prim)


def _create_test_mesh_usd(path):
    """Create a small USD mesh asset for tests."""
    stage = Usd.Stage.CreateNew(str(path))
    root_prim = stage.DefinePrim("/ConvertedMesh", "Xform")
    stage.SetDefaultPrim(root_prim)
    UsdGeom.Xform.Define(stage, "/ConvertedMesh/geometry")
    mesh_prim = UsdGeom.Mesh.Define(stage, "/ConvertedMesh/geometry/mesh")
    mesh_prim.GetPointsAttr().Set(_TRIANGLE_VERTICES)
    mesh_prim.GetFaceVertexCountsAttr().Set([3])
    mesh_prim.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    stage.GetRootLayer().Save()


@pytest.mark.isaacsim_ci
def test_spawn_mesh_file_with_asset_path(sim, tmp_path, monkeypatch):
    """Test spawning a mesh file path through the mesh converter path."""
    import isaaclab.sim.spawners.from_files.from_files as from_files_module

    obj_path = tmp_path / "triangle.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    converted_usd_path = tmp_path / "triangle.usd"
    _create_test_mesh_usd(converted_usd_path)
    captured = {}

    class FakeMeshConverter:
        def __init__(self, cfg):
            captured["cfg"] = cfg
            self.usd_path = str(converted_usd_path)

    monkeypatch.setattr(from_files_module.converters, "MeshConverter", FakeMeshConverter)

    cfg = sim_utils.MeshFileCfg(
        mesh=str(obj_path),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=sim_utils.ConvexHullPropertiesCfg(),
    )

    prim = cfg.func("/World/FileMesh", cfg)

    assert prim.IsValid()
    mesh_prim = sim.stage.GetPrimAtPath("/World/FileMesh/geometry/mesh")
    assert mesh_prim.IsValid()
    assert captured["cfg"].asset_path == str(obj_path)
    assert captured["cfg"].collision_props == cfg.collision_props
    assert captured["cfg"].mesh_collision_props == cfg.mesh_collision_props


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("parent_suffix", ["", "/Terrain"])
def test_spawn_mesh_file_clones_regex_parent_with_suffix(sim, parent_suffix):
    """Test regex cloning with existing and newly-created concrete parent suffixes."""
    sim_utils.create_prim(f"/World/envs/env_0{parent_suffix}", "Xform")
    sim_utils.create_prim(f"/World/envs/env_1{parent_suffix}", "Xform")
    cfg = _triangle_mesh_cfg()

    prim = cfg.func("/World/envs/env_.*/Terrain/Ground", cfg)

    assert prim.GetPath().pathString == "/World/envs/env_0/Terrain/Ground"
    assert sim.stage.GetPrimAtPath("/World/envs/env_0/Terrain/Ground/mesh").IsValid()
    assert sim.stage.GetPrimAtPath("/World/envs/env_1/Terrain/Ground/mesh").IsValid()


@pytest.mark.isaacsim_ci
def test_spawn_mesh_file_replaces_leaf_regex(sim):
    """Test leaf regex replacement keeps the existing clone behavior."""
    sim_utils.create_prim("/World/template/Object", "Xform")
    cfg = _triangle_mesh_cfg()

    prim = cfg.func("/World/template/Object/proto_asset_.*", cfg)

    assert prim.GetPath().pathString == "/World/template/Object/proto_asset_0"
    assert sim.stage.GetPrimAtPath("/World/template/Object/proto_asset_0/mesh").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/template/Object/proto_asset_.*/mesh").IsValid()


@pytest.mark.isaacsim_ci
def test_spawn_mesh_file_does_not_create_regex_suffix(sim):
    """Test clone fallback does not create unmatched regex suffix paths."""
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    cfg = _triangle_mesh_cfg()

    with pytest.raises(RuntimeError):
        cfg.func("/World/envs/env_.*/Terrain_.*/Ground", cfg)


@pytest.mark.isaacsim_ci
def test_create_prim_from_mesh_uses_triangle_collision(sim):
    """Test terrain mesh helper keeps the historical triangle-mesh collider."""
    from isaaclab.terrains.utils import create_prim_from_mesh

    prim = create_prim_from_mesh("/World/Terrain", trimesh.creation.box())

    assert prim.IsValid()
    mesh_prim = sim.stage.GetPrimAtPath("/World/Terrain/mesh")
    assert mesh_prim.IsValid()
    assert UsdPhysics.CollisionAPI(mesh_prim).GetCollisionEnabledAttr().Get() is True
    assert UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get() == MESH_APPROXIMATION_TOKENS["none"]


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
