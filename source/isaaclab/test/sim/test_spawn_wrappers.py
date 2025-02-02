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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


class TestSpawningWrappers(unittest.TestCase):
    """Test fixture for checking spawning of multiple assets wrappers."""

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.1
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")
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
    Tests - Multiple assets.
    """

    def test_spawn_multiple_shapes_with_global_settings(self):
        """Test spawning of shapes randomly with global rigid body settings."""
        # Define prim parents
        num_clones = 10
        for i in range(num_clones):
            prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

        # Spawn shapes
        cfg = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),  # this one should get overridden
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        prim = cfg.func("/World/env_.*/Cone", cfg)

        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim_utils.get_prim_path(prim), "/World/env_0/Cone")
        # Find matching prims
        prim_paths = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
        self.assertEqual(len(prim_paths), num_clones)

        # Check all prims have correct settings
        for prim_path in prim_paths:
            prim = prim_utils.get_prim_at_path(prim_path)
            self.assertEqual(prim.GetAttribute("physics:mass").Get(), cfg.mass_props.mass)

    def test_spawn_multiple_shapes_with_individual_settings(self):
        """Test spawning of shapes randomly with individual rigid object settings"""
        # Define prim parents
        num_clones = 10
        for i in range(num_clones):
            prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

        # Make a list of masses
        mass_variations = [2.0, 3.0, 4.0]
        # Spawn shapes
        cfg = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass_variations[0]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass_variations[1]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass_variations[2]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
            ],
            random_choice=True,
        )
        prim = cfg.func("/World/env_.*/Cone", cfg)

        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim_utils.get_prim_path(prim), "/World/env_0/Cone")
        # Find matching prims
        prim_paths = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
        self.assertEqual(len(prim_paths), num_clones)

        # Check all prims have correct settings
        for prim_path in prim_paths:
            prim = prim_utils.get_prim_at_path(prim_path)
            self.assertTrue(prim.GetAttribute("physics:mass").Get() in mass_variations)

    """
    Tests - Multiple USDs.
    """

    def test_spawn_multiple_files_with_global_settings(self):
        """Test spawning of files randomly with global articulation settings."""
        # Define prim parents
        num_clones = 10
        for i in range(num_clones):
            prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

        # Spawn shapes
        cfg = sim_utils.MultiUsdFileCfg(
            usd_path=[
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            activate_contact_sensors=True,
        )
        prim = cfg.func("/World/env_.*/Robot", cfg)

        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim_utils.get_prim_path(prim), "/World/env_0/Robot")
        # Find matching prims
        prim_paths = prim_utils.find_matching_prim_paths("/World/env_*/Robot")
        self.assertEqual(len(prim_paths), num_clones)


if __name__ == "__main__":
    run_tests()
