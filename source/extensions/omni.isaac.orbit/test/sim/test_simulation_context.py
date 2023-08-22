# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext as IsaacSimulationContext

from omni.isaac.orbit.sim import SimulationCfg, SimulationContext


class TestSimulationContext(unittest.TestCase):
    """Test fixture for wrapper around simulation context."""

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Load kit helper
        SimulationContext.clear_instance()

    def test_singleton(self):
        """Tests that the singleton is working."""
        sim1 = SimulationContext()
        sim2 = SimulationContext()
        sim3 = IsaacSimulationContext()
        self.assertIs(sim1, sim2)
        self.assertIs(sim1, sim3)

        # try to delete the singleton
        sim2.__del__()
        self.assertTrue(sim1.instance() is None)
        # create new instance
        sim4 = SimulationContext()
        self.assertIsNot(sim1, sim4)
        self.assertIsNot(sim3, sim4)
        self.assertIs(sim1.instance(), sim4.instance())
        self.assertIs(sim3.instance(), sim4.instance())
        # clear instance
        sim3.clear_instance()

    def test_initialization(self):
        """Test the simulation config."""
        cfg = SimulationCfg(physics_prim_path="/Physics/PhysX", substeps=5, gravity=(0.0, -0.5, -0.5))
        sim = SimulationContext(cfg)
        # TODO: Figure out why keyword argument doesn't work.
        # note: added a fix in Isaac Sim 2023.1 for this.
        # sim = SimulationContext(cfg=cfg)

        # check valid settings
        self.assertEqual(sim.get_physics_dt(), cfg.dt)
        self.assertEqual(sim.get_rendering_dt(), cfg.dt * cfg.substeps)
        # check valid paths
        self.assertTrue(prim_utils.is_prim_path_valid("/Physics/PhysX"))
        self.assertTrue(prim_utils.is_prim_path_valid("/Physics/PhysX/defaultMaterial"))
        # check valid gravity
        gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
        gravity = np.array(gravity_dir) * gravity_mag
        np.testing.assert_almost_equal(gravity, cfg.gravity)

    def test_sim_version(self):
        """Test obtaining the version."""
        sim = SimulationContext()
        version = sim.get_version()
        self.assertTrue(len(version) > 0)
        self.assertTrue(version[0] >= 2022)

    def test_carb_setting(self):
        """Test setting carb settings."""
        sim = SimulationContext()
        # known carb setting
        sim.set_setting("/physics/physxDispatcher", False)
        self.assertEqual(sim.get_setting("/physics/physxDispatcher"), False)
        # unknown carb setting
        sim.set_setting("/myExt/using_omniverse_version", sim.get_version())
        self.assertSequenceEqual(sim.get_setting("/myExt/using_omniverse_version"), sim.get_version())


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
