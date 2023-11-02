# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.kit import SimulationApp

# launch the simulator
config = {"headless": False}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.compat.utils.kit as kit_utils
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


class TestKitUtilities(unittest.TestCase):
    """Test fixture for checking Kit utilities in Orbit."""

    @classmethod
    def tearDownClass(cls):
        """Closes simulator after running all test fixtures."""
        simulation_app.close()

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Simulation time-step
        self.dt = 0.1
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")
        # Set camera view
        set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        # Spawn things into stage
        self._populate_scene()
        # Wait for spawning
        stage_utils.update_stage()

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()

    def test_rigid_body_properties(self):
        """Disable setting of rigid body properties."""
        # create marker
        prim_utils.create_prim(
            "/World/marker", usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        )
        # set marker properties
        kit_utils.set_nested_rigid_body_properties("/World/marker", rigid_body_enabled=False)
        kit_utils.set_nested_collision_properties("/World/marker", collision_enabled=False)

        # play simulation
        self.sim.reset()
        for _ in range(5):
            self.sim.step()

    """
    Helper functions.
    """

    @staticmethod
    def _populate_scene():
        """Add prims to the scene."""
        # Ground-plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane")
        # Lights-1
        prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
        # Lights-2
        prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))


if __name__ == "__main__":
    unittest.main()
