# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math
import os
import random
import tempfile
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni
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


class TestMeshConverter(unittest.TestCase):
    """Test fixture for the MeshConverter class."""

    @classmethod
    def setUpClass(cls):
        """Load assets for tests."""
        assets_dir = f"{ISAACLAB_NUCLEUS_DIR}/Tests/MeshConverter/duck"
        # Create mapping of file endings to file paths that can be used by tests
        cls.assets = {
            "obj": f"{assets_dir}/duck.obj",
            "stl": f"{assets_dir}/duck.stl",
            "fbx": f"{assets_dir}/duck.fbx",
            "mtl": f"{assets_dir}/duck.mtl",
            "png": f"{assets_dir}/duckCM.png",
        }
        # Download all these locally
        download_dir = tempfile.mkdtemp(suffix="_mesh_converter_test_assets")
        for key, value in cls.assets.items():
            cls.assets[key] = retrieve_file_path(value, download_dir=download_dir)

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # cleanup stage and context
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Test fixtures.
    """

    def test_no_change(self):
        """Call conversion twice on the same input asset. This should not generate a new USD file if the hash is the same."""
        # create an initial USD file from asset
        mesh_config = MeshConverterCfg(asset_path=self.assets["obj"])
        mesh_converter = MeshConverter(mesh_config)
        time_usd_file_created = os.stat(mesh_converter.usd_path).st_mtime_ns

        # no change to config only define the usd directory
        new_config = mesh_config
        new_config.usd_dir = mesh_converter.usd_dir
        # convert to usd but this time in the same directory as previous step
        new_mesh_converter = MeshConverter(new_config)
        new_time_usd_file_created = os.stat(new_mesh_converter.usd_path).st_mtime_ns

        self.assertEqual(time_usd_file_created, new_time_usd_file_created)

    def test_config_change(self):
        """Call conversion twice but change the config in the second call. This should generate a new USD file."""
        # create an initial USD file from asset
        mesh_config = MeshConverterCfg(asset_path=self.assets["obj"])
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

        self.assertNotEqual(time_usd_file_created, new_time_usd_file_created)

    def test_convert_obj(self):
        """Convert an OBJ file"""
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            scale=(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)),
            translation=(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)),
            rotation=random_quaternion(),
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_conversion(mesh_converter)

    def test_convert_stl(self):
        """Convert an STL file"""
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["stl"],
            scale=(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)),
            translation=(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)),
            rotation=random_quaternion(),
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_conversion(mesh_converter)

    def test_convert_fbx(self):
        """Convert an FBX file"""
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["fbx"],
            scale=(random.uniform(0.1, 2.0), random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)),
            translation=(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)),
            rotation=random_quaternion(),
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_conversion(mesh_converter)

    def test_convert_default_xform_transforms(self):
        """Convert an OBJ file and check that default xform transforms are applied correctly"""
        mesh_config = MeshConverterCfg(asset_path=self.assets["obj"])
        mesh_converter = MeshConverter(mesh_config)
        # check that mesh conversion is successful
        self._check_mesh_conversion(mesh_converter)

    def test_collider_no_approximation(self):
        """Convert an OBJ file using no approximation"""
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            collision_approximation="none",
            collision_props=collision_props,
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_collider_settings(mesh_converter)

    def test_collider_convex_hull(self):
        """Convert an OBJ file using convex hull approximation"""
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            collision_approximation="convexHull",
            collision_props=collision_props,
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_collider_settings(mesh_converter)

    def test_collider_mesh_simplification(self):
        """Convert an OBJ file using mesh simplification approximation"""
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            collision_approximation="meshSimplification",
            collision_props=collision_props,
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_collider_settings(mesh_converter)

    def test_collider_mesh_bounding_cube(self):
        """Convert an OBJ file using bounding cube approximation"""
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            collision_approximation="boundingCube",
            collision_props=collision_props,
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_collider_settings(mesh_converter)

    def test_collider_mesh_bounding_sphere(self):
        """Convert an OBJ file using bounding sphere"""
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            collision_approximation="boundingSphere",
            collision_props=collision_props,
        )
        mesh_converter = MeshConverter(mesh_config)

        # check that mesh conversion is successful
        self._check_mesh_collider_settings(mesh_converter)

    def test_collider_mesh_no_collision(self):
        """Convert an OBJ file using bounding sphere with collision disabled"""
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=False)
        mesh_config = MeshConverterCfg(
            asset_path=self.assets["obj"],
            collision_approximation="boundingSphere",
            collision_props=collision_props,
        )
        mesh_converter = MeshConverter(mesh_config)
        # check that mesh conversion is successful
        self._check_mesh_collider_settings(mesh_converter)

    """
    Helper functions.
    """

    def _check_mesh_conversion(self, mesh_converter: MeshConverter):
        """Check that mesh is loadable and stage is valid."""
        # Load the mesh
        prim_path = "/World/Object"
        prim_utils.create_prim(prim_path, usd_path=mesh_converter.usd_path)
        # Check prim can be properly spawned
        self.assertTrue(prim_utils.is_prim_path_valid(prim_path))
        # Load a second time
        prim_path = "/World/Object2"
        prim_utils.create_prim(prim_path, usd_path=mesh_converter.usd_path)
        # Check prim can be properly spawned
        self.assertTrue(prim_utils.is_prim_path_valid(prim_path))

        stage = omni.usd.get_context().get_stage()
        # Check axis is z-up
        axis = UsdGeom.GetStageUpAxis(stage)
        self.assertEqual(axis, "Z")
        # Check units is meters
        units = UsdGeom.GetStageMetersPerUnit(stage)
        self.assertEqual(units, 1.0)

        # Check mesh settings
        pos = tuple(prim_utils.get_prim_at_path("/World/Object/geometry").GetAttribute("xformOp:translate").Get())
        self.assertEqual(pos, mesh_converter.cfg.translation)
        quat = prim_utils.get_prim_at_path("/World/Object/geometry").GetAttribute("xformOp:orient").Get()
        quat = (quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2])
        self.assertEqual(quat, mesh_converter.cfg.rotation)
        scale = tuple(prim_utils.get_prim_at_path("/World/Object/geometry").GetAttribute("xformOp:scale").Get())
        self.assertEqual(scale, mesh_converter.cfg.scale)

    def _check_mesh_collider_settings(self, mesh_converter: MeshConverter):
        # Check prim can be properly spawned
        prim_path = "/World/Object"
        prim_utils.create_prim(prim_path, usd_path=mesh_converter.usd_path)
        self.assertTrue(prim_utils.is_prim_path_valid(prim_path))

        # Make uninstanceable to check collision settings
        geom_prim = prim_utils.get_prim_at_path(prim_path + "/geometry")
        # Check that instancing worked!
        self.assertEqual(geom_prim.IsInstanceable(), mesh_converter.cfg.make_instanceable)
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
        self.assertEqual(collision_enabled, exp_collision_enabled, "Collision enabled is not the same!")
        # -- if collision is enabled, check that collision approximation is correct
        if exp_collision_enabled:
            exp_collision_approximation = mesh_converter.cfg.collision_approximation
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(mesh_prim)
            collision_approximation = mesh_collision_api.GetApproximationAttr().Get()
            self.assertEqual(
                collision_approximation, exp_collision_approximation, "Collision approximation is not the same!"
            )


if __name__ == "__main__":
    run_tests()
