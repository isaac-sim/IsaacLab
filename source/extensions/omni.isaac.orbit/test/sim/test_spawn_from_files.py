# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.app import AppLauncher

"""Launch Isaac Sim Simulator first."""

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.extensions import get_extension_path_from_name

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR


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
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
        prim = cfg.func("/World/Franka", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/Franka"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Xform")

    def test_spawn_urdf(self):
        """Test loading prim from URDF file."""
        # retrieve path to urdf importer extension
        extension_path = get_extension_path_from_name("omni.importer.urdf")
        # Spawn franka from URDF
        cfg = sim_utils.UrdfFileCfg(
            asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf", fix_base=True
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
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
