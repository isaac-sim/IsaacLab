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
import os
import random
import tempfile

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni
import pytest
from isaacsim.core.api.simulation_context import SimulationContext
from pxr import UsdGeom, UsdPhysics

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path


def random_quaternion():
    # Generate four random numbers for the quaternion
    u1, u2, u3 = random.random(), random.random(), random.random()
    w = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    x = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    y = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    z = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return (w, x, y, z)


@pytest.fixture(scope="session")
def assets():
    """Load assets for tests."""
    assets_dir = f"{ISAACLAB_NUCLEUS_DIR}/Tests/MeshConverter/duck"
    # Create mapping of file endings to file paths that can be used by tests
    assets = {
        "obj": f"{assets_dir}/duck.obj",
        "stl": f"{assets_dir}/duck.stl",
        "fbx": f"{assets_dir}/duck.fbx",
        "mtl": f"{assets_dir}/duck.mtl",
        "png": f"{assets_dir}/duckCM.png",
    }
    # Download all these locally
    download_dir = tempfile.mkdtemp(suffix="_mesh_converter_test_assets")
    for key, value in assets.items():
        assets[key] = retrieve_file_path(value, download_dir=download_dir)
    return assets


@pytest.fixture(autouse=True)
def sim():
    """Create a blank new stage for each test."""
    # Create a new stage
    stage_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="numpy")
    yield sim
    # stop simulation
    sim.stop()
    # cleanup stage and context
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


def check_mesh_conversion(mesh_converter: MeshConverter):
    """Check that mesh is loadable and stage is valid."""
    # Load the mesh
    prim_path = "/World/Object"
    prim_utils.create_prim(prim_path, usd_path=mesh_converter.usd_path)
    # Check prim can be properly spawned
    assert prim_utils.is_prim_path_valid(prim_path)
    # Load a second time
    prim_path = "/World/Object2"
    prim_utils.create_prim(prim_path, usd_path=mesh_converter.usd_path)
    # Check prim can be properly spawned
    assert prim_utils.is_prim_path_valid(prim_path)

    stage = omni.usd.get_context().get_stage()
    # Check axis is z-up
    axis = UsdGeom.GetStageUpAxis(stage)
    assert axis == "Z"
    # Check units is meters
    units = UsdGeom.GetStageMetersPerUnit(stage)
    assert units == 1.0

    # Check mesh settings
    pos = tuple(prim_utils.get_prim_at_path("/World/Object/geometry").GetAttribute("xformOp:translate").Get())
    assert pos == mesh_converter.cfg.translation
    quat = prim_utils.get_prim_at_path("/World/Object/geometry").GetAttribute("xformOp:orient").Get()
    quat = (quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2])
    assert quat == mesh_converter.cfg.rotation
    scale = tuple(prim_utils.get_prim_at_path("/World/Object/geometry").GetAttribute("xformOp:scale").Get())
    assert scale == mesh_converter.cfg.scale


def check_mesh_collider_settings(mesh_converter: MeshConverter):
    """Check that mesh collider settings are correct."""
    # Check prim can be properly spawned
    prim_path = "/World/Object"
    prim_utils.create_prim(prim_path, usd_path=mesh_converter.usd_path)
    assert prim_utils.is_prim_path_valid(prim_path)

    # Make uninstanceable to check collision settings
    geom_prim = prim_utils.get_prim_at_path(prim_path + "/geometry")
    # Check that instancing worked!
    assert geom_prim.IsInstanceable() == mesh_converter.cfg.make_instanceable
    # Obtain mesh settings
    geom_prim.SetInstanceable(False)
    mesh_prim = prim_utils.get_prim_at_path(prim_path + "/geometry/mesh")

    # Check collision settings
    # -- if collision is enabled, check that API is present
    exp_collision_enabled = (
        mesh_converter.cfg.collision_props is not None and mesh_converter.cfg.collision_props.collision_enabled
    )
    collision_api = UsdPhysics.CollisionAPI(mesh_prim)
    collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
    assert collision_enabled == exp_collision_enabled, "Collision enabled is not the same!"
    # -- if collision is enabled, check that collision approximation is correct
    if exp_collision_enabled:
        exp_collision_approximation = mesh_converter.cfg.collision_approximation
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(mesh_prim)
        collision_approximation = mesh_collision_api.GetApproximationAttr().Get()
        assert collision_approximation == exp_collision_approximation, "Collision approximation is not the same!"


def test_no_change(assets):
    """Call conversion twice on the same input asset. This should not generate a new USD file if the hash is the same."""
    # create an initial USD file from asset
    mesh_config = MeshConverterCfg(asset_path=assets["obj"])
    mesh_converter = MeshConverter(mesh_config)
    time_usd_file_created = os.stat(mesh_converter.usd_path).st_mtime_ns

    # no change to config only define the usd directory
    new_config = mesh_config
    new_config.usd_dir = mesh_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_mesh_converter = MeshConverter(new_config)
    new_time_usd_file_created = os.stat(new_mesh_converter.usd_path).st_mtime_ns

    assert time_usd_file_created == new_time_usd_file_created


def test_config_change(assets):
    """Call conversion twice but change the config in the second call. This should generate a new USD file."""
    # create an initial USD file from asset
    mesh_config = MeshConverterCfg(asset_path=assets["obj"])
    mesh_converter = MeshConverter(mesh_config)
    time_usd_file_created = os.stat(mesh_converter.usd_path).st_mtime_ns

    # change the config
    new_config = mesh_config
    new_config.make_instanceable = not mesh_config.make_instanceable
    # define the usd directory
    new_config.usd_dir = mesh_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_mesh_converter = MeshConverter(new_config)
    new_time_usd_file_created = os.stat(new_mesh_converter.usd_path).st_mtime_ns

    assert time_usd_file_created != new_time_usd_file_created


def test_convert_obj(assets):
    """Convert an OBJ file"""
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        scale=(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)),
        translation=(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)),
        rotation=random_quaternion(),
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_conversion(mesh_converter)


def test_convert_stl(assets):
    """Convert an STL file"""
    mesh_config = MeshConverterCfg(
        asset_path=assets["stl"],
        scale=(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)),
        translation=(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)),
        rotation=random_quaternion(),
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_conversion(mesh_converter)


def test_convert_fbx(assets):
    """Convert an FBX file"""
    mesh_config = MeshConverterCfg(
        asset_path=assets["fbx"],
        scale=(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)),
        translation=(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)),
        rotation=random_quaternion(),
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_conversion(mesh_converter)


def test_convert_default_xform_transforms(assets):
    """Convert an OBJ file and check that default xform transforms are applied correctly"""
    mesh_config = MeshConverterCfg(asset_path=assets["obj"])
    mesh_converter = MeshConverter(mesh_config)
    # check that mesh conversion is successful
    check_mesh_conversion(mesh_converter)


def test_collider_no_approximation(assets):
    """Convert an OBJ file using no approximation"""
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_approximation="none",
        collision_props=collision_props,
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_collider_settings(mesh_converter)


def test_collider_convex_hull(assets):
    """Convert an OBJ file using convex hull approximation"""
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_approximation="convexHull",
        collision_props=collision_props,
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_collider_settings(mesh_converter)


def test_collider_mesh_simplification(assets):
    """Convert an OBJ file using mesh simplification approximation"""
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_approximation="meshSimplification",
        collision_props=collision_props,
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_collider_settings(mesh_converter)


def test_collider_mesh_bounding_cube(assets):
    """Convert an OBJ file using bounding cube approximation"""
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_approximation="boundingCube",
        collision_props=collision_props,
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_collider_settings(mesh_converter)


def test_collider_mesh_bounding_sphere(assets):
    """Convert an OBJ file using bounding sphere"""
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_approximation="boundingSphere",
        collision_props=collision_props,
    )
    mesh_converter = MeshConverter(mesh_config)

    # check that mesh conversion is successful
    check_mesh_collider_settings(mesh_converter)


def test_collider_mesh_no_collision(assets):
    """Convert an OBJ file using bounding sphere with collision disabled"""
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=False)
    mesh_config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_approximation="boundingSphere",
        collision_props=collision_props,
    )
    mesh_converter = MeshConverter(mesh_config)
    # check that mesh conversion is successful
    check_mesh_collider_settings(mesh_converter)
