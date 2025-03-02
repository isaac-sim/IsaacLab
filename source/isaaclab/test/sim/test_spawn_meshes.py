# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.simulation_context import SimulationContext

import isaaclab.sim as sim_utils


class TestSpawningMeshGeometries(unittest.TestCase):
    """Test fixture for checking spawning of USD-Mesh prim with different settings."""

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.1
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, device="cuda:0")
        # Wait for spawning
        stage_utils.update_stage()

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Basic spawning.
    """

    def test_spawn_cone(self):
        """Test spawning of UsdGeomMesh as a cone prim."""
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(radius=1.0, height=2.0, axis="Y")
        prim = cfg.func("/World/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Mesh")

    def test_spawn_capsule(self):
        """Test spawning of UsdGeomMesh as a capsule prim."""
        # Spawn capsule
        cfg = sim_utils.MeshCapsuleCfg(radius=1.0, height=2.0, axis="Y")
        prim = cfg.func("/World/Capsule", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Capsule"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Capsule/geometry/mesh")
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Mesh")

    def test_spawn_cylinder(self):
        """Test spawning of UsdGeomMesh as a cylinder prim."""
        # Spawn cylinder
        cfg = sim_utils.MeshCylinderCfg(radius=1.0, height=2.0, axis="Y")
        prim = cfg.func("/World/Cylinder", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cylinder"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Cylinder/geometry/mesh")
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Mesh")

    def test_spawn_cuboid(self):
        """Test spawning of UsdGeomMesh as a cuboid prim."""
        # Spawn cuboid
        cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0))
        prim = cfg.func("/World/Cube", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cube"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Cube/geometry/mesh")
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Mesh")

    def test_spawn_sphere(self):
        """Test spawning of UsdGeomMesh as a sphere prim."""
        # Spawn sphere
        cfg = sim_utils.MeshSphereCfg(radius=1.0)
        prim = cfg.func("/World/Sphere", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Sphere"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Sphere/geometry/mesh")
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Mesh")

    """
    Physics properties.
    """

    def test_spawn_cone_with_deformable_props(self):
        """Test spawning of UsdGeomMesh prim for a cone with deformable body API."""
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(
            radius=1.0,
            height=2.0,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
        )
        prim = cfg.func("/World/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone"))

        # Check properties
        # Unlike rigid body, deformable body properties are on the mesh prim
        prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
        self.assertEqual(
            prim.GetAttribute("physxDeformable:deformableEnabled").Get(), cfg.deformable_props.deformable_enabled
        )

    def test_spawn_cone_with_deformable_and_mass_props(self):
        """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API."""
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(
            radius=1.0,
            height=2.0,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        )
        prim = cfg.func("/World/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone"))
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
        self.assertEqual(prim.GetAttribute("physics:mass").Get(), cfg.mass_props.mass)

        # check sim playing
        self.sim.play()
        for _ in range(10):
            self.sim.step()

    def test_spawn_cone_with_deformable_and_density_props(self):
        """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API.

        Note:
            In this case, we specify the density instead of the mass. In that case, physics need to know
            the collision shape to compute the mass. Thus, we have to set the collider properties. In
            order to not have a collision shape, we disable the collision.
        """
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(
            radius=1.0,
            height=2.0,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(density=10.0),
        )
        prim = cfg.func("/World/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone"))
        # Check properties
        prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
        self.assertEqual(prim.GetAttribute("physics:density").Get(), cfg.mass_props.density)
        # check sim playing
        self.sim.play()
        for _ in range(10):
            self.sim.step()

    def test_spawn_cone_with_all_deformable_props(self):
        """Test spawning of UsdGeomMesh prim for a cone with all deformable properties."""
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(
            radius=1.0,
            height=2.0,
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
            physics_material=sim_utils.materials.DeformableBodyMaterialCfg(),
        )
        prim = cfg.func("/World/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone"))
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone/geometry/material"))
        # Check properties
        # -- deformable body
        prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
        self.assertEqual(prim.GetAttribute("physxDeformable:deformableEnabled").Get(), True)

        # check sim playing
        self.sim.play()
        for _ in range(10):
            self.sim.step()

    def test_spawn_cone_with_all_rigid_props(self):
        """Test spawning of UsdGeomMesh prim for a cone with all rigid properties."""
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(
            radius=1.0,
            height=2.0,
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(),
        )
        prim = cfg.func("/World/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone"))
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Cone/geometry/material"))
        # Check properties
        # -- rigid body
        prim = prim_utils.get_prim_at_path("/World/Cone")
        self.assertEqual(prim.GetAttribute("physics:rigidBodyEnabled").Get(), cfg.rigid_props.rigid_body_enabled)
        self.assertEqual(
            prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get(),
            cfg.rigid_props.solver_position_iteration_count,
        )
        self.assertAlmostEqual(
            prim.GetAttribute("physxRigidBody:sleepThreshold").Get(), cfg.rigid_props.sleep_threshold
        )
        # -- mass
        self.assertEqual(prim.GetAttribute("physics:mass").Get(), cfg.mass_props.mass)
        # -- collision shape
        prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
        self.assertEqual(prim.GetAttribute("physics:collisionEnabled").Get(), True)

        # check sim playing
        self.sim.play()
        for _ in range(10):
            self.sim.step()

    def test_spawn_deformable_rigid_cone_invalid(self):
        """Test specifying both rigid and deformable properties for a cone causes an error."""
        # Spawn cone
        with self.assertRaises(ValueError):
            cfg = sim_utils.MeshConeCfg(
                radius=1.0,
                height=2.0,
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
            )
            cfg.func("/World/Cone", cfg)

    def test_spawn_deformable_collider_cone_invalid(self):
        """Test specifying both deformable and collider properties for a cone causes an error."""
        # Spawn cone
        with self.assertRaises(ValueError):
            cfg = sim_utils.MeshConeCfg(
                radius=1.0,
                height=2.0,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
            )
            cfg.func("/World/Cone", cfg)

    def test_spawn_deformable_incorrect_material(self):
        """Test specifying incorrect material for deformable object causes an error."""
        # Spawn cone
        with self.assertRaises(ValueError):
            cfg = sim_utils.MeshConeCfg(
                radius=1.0,
                height=2.0,
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
                visual_material=sim_utils.materials.PreviewSurfaceCfg(),
                physics_material=sim_utils.materials.RigidBodyMaterialCfg(),
            )
            cfg.func("/World/Cone", cfg)

    def test_spawn_rigid_incorrect_material(self):
        """Test specifying incorrect material for rigid object causes an error."""
        # Spawn cone
        with self.assertRaises(ValueError):
            cfg = sim_utils.MeshConeCfg(
                radius=1.0,
                height=2.0,
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                visual_material=sim_utils.materials.PreviewSurfaceCfg(),
                physics_material=sim_utils.materials.DeformableBodyMaterialCfg(),
            )
            cfg.func("/World/Cone", cfg)

    """
    Cloning.
    """

    def test_spawn_cone_clones_invalid_paths(self):
        """Test spawning of cone clones on invalid cloning paths."""
        num_clones = 10
        for i in range(num_clones):
            prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(radius=1.0, height=2.0, copy_from_source=True)
        # Should raise error for invalid path
        with self.assertRaises(RuntimeError):
            cfg.func("/World/env/env_.*/Cone", cfg)

    def test_spawn_cone_clones(self):
        """Test spawning of cone clones."""
        num_clones = 10
        for i in range(num_clones):
            prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(radius=1.0, height=2.0, copy_from_source=True)
        prim = cfg.func("/World/env_.*/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim_utils.get_prim_path(prim), "/World/env_0/Cone")
        # Find matching prims
        prims = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
        self.assertEqual(len(prims), num_clones)

    def test_spawn_cone_clone_with_all_props_global_material(self):
        """Test spawning of cone clones with global material reference."""
        num_clones = 10
        for i in range(num_clones):
            prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
        # Spawn cone
        cfg = sim_utils.MeshConeCfg(
            radius=1.0,
            height=2.0,
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
            physics_material=sim_utils.materials.DeformableBodyMaterialCfg(),
            visual_material_path="/Looks/visualMaterial",
            physics_material_path="/Looks/physicsMaterial",
        )
        prim = cfg.func("/World/env_.*/Cone", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim_utils.get_prim_path(prim), "/World/env_0/Cone")
        # Find matching prims
        prims = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
        self.assertEqual(len(prims), num_clones)
        # Find global materials
        prims = prim_utils.find_matching_prim_paths("/Looks/visualMaterial.*")
        self.assertEqual(len(prims), 1)


if __name__ == "__main__":
    run_tests()
