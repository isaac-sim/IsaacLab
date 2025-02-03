# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher, run_tests

"""Launch Isaac Sim Simulator first."""

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


class TestSpawningFromFiles(unittest.TestCase):
    """Test fixture for checking spawning of USD references from files with different settings."""

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
    Basic spawning.
    """

    def test_spawn_usd(self):
        """Test loading prim from Usd file."""
        # Spawn cone
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
        prim = cfg.func("/World/Franka", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Franka"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")

    def test_spawn_usd_fails(self):
        """Test loading prim from Usd file fails when asset usd path is invalid."""
        # Spawn cone
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda2_instanceable.usd")

        with self.assertRaises(FileNotFoundError):
            cfg.func("/World/Franka", cfg)

    def test_spawn_urdf(self):
        """Test loading prim from URDF file."""
        # retrieve path to urdf importer extension
        enable_extension("isaacsim.asset.importer.urdf")
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
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
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Franka"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")

    def test_spawn_ground_plane(self):
        """Test loading prim for the ground plane from grid world USD."""
        # Spawn ground plane
        cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(10.0, 10.0))
        prim = cfg.func("/World/ground_plane", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/ground_plane"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")


if __name__ == "__main__":
    run_tests()
