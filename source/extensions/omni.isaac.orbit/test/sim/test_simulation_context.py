# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import ctypes
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
        sim2.clear_instance()
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
        self.assertTrue(version[0] >= 2023)

    def test_carb_setting(self):
        """Test setting carb settings."""
        sim = SimulationContext()
        # known carb setting
        sim.set_setting("/physics/physxDispatcher", False)
        self.assertEqual(sim.get_setting("/physics/physxDispatcher"), False)
        # unknown carb setting
        sim.set_setting("/myExt/using_omniverse_version", sim.get_version())
        self.assertSequenceEqual(sim.get_setting("/myExt/using_omniverse_version"), sim.get_version())

    def test_headless_mode(self):
        """Test that render mode is headless since we are running in headless mode."""

        sim = SimulationContext()
        # check default render mode
        self.assertEqual(sim.render_mode, sim.RenderMode.NO_GUI_OR_RENDERING)

    def test_boundedness(self):
        """Test that the boundedness of the simulation context remains constant.

        Note: This test fails right now because Isaac Sim does not handle boundedness correctly. On creation,
        it is registering itself to various callbacks and hence the boundedness is more than 1. This may not be
        critical for the simulation context since we usually call various clear functions before deleting the
        simulation context.
        """
        sim = SimulationContext()
        # manually set the boundedness to 1? -- this is not possible because of Isaac Sim.
        sim.clear_all_callbacks()
        sim._stage_open_callback = None
        sim._physics_timer_callback = None
        sim._event_timer_callback = None

        # check that boundedness of simulation context is correct
        sim_ref_count = ctypes.c_long.from_address(id(sim)).value
        # reset the simulation
        sim.reset()
        self.assertEqual(ctypes.c_long.from_address(id(sim)).value, sim_ref_count)
        # step the simulation
        for _ in range(10):
            sim.step()
            self.assertEqual(ctypes.c_long.from_address(id(sim)).value, sim_ref_count)
        # clear the simulation
        sim.clear_instance()
        self.assertEqual(ctypes.c_long.from_address(id(sim)).value, sim_ref_count - 1)

    def test_zero_gravity(self):
        """Test that gravity can be properly disabled."""
        cfg = SimulationCfg(gravity=(0.0, 0.0, 0.0))

        sim = SimulationContext(cfg)

        gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
        gravity = np.array(gravity_dir) * gravity_mag
        np.testing.assert_almost_equal(gravity, cfg.gravity)


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
