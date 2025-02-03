# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This test has a lot of duplication with ``test_build_simulation_context_headless.py``. This is intentional to ensure that the
tests are run in both headless and non-headless modes, and we currently can't re-build the simulation app in a script.

If you need to make a change to this test, please make sure to also make the same change to ``test_build_simulation_context_headless.py``.

"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import unittest

from isaacsim.core.utils.prims import is_prim_path_valid

from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import build_simulation_context


class TestBuildSimulationContextNonheadless(unittest.TestCase):
    """Tests for simulation context builder with non-headless usecase."""

    """
    Tests
    """

    def test_build_simulation_context_no_cfg(self):
        """Test that the simulation context is built when no simulation cfg is passed in."""
        for gravity_enabled in (True, False):
            for device in ("cuda:0", "cpu"):
                for dt in (0.01, 0.1):
                    with self.subTest(gravity_enabled=gravity_enabled, device=device, dt=dt):
                        with build_simulation_context(
                            gravity_enabled=gravity_enabled,
                            device=device,
                            dt=dt,
                        ) as sim:
                            if gravity_enabled:
                                self.assertEqual(sim.cfg.gravity, (0.0, 0.0, -9.81))
                            else:
                                self.assertEqual(sim.cfg.gravity, (0.0, 0.0, 0.0))

                            if device == "cuda:0":
                                self.assertEqual(sim.cfg.device, "cuda:0")
                            else:
                                self.assertEqual(sim.cfg.device, "cpu")

                            self.assertEqual(sim.cfg.dt, dt)

    def test_build_simulation_context_ground_plane(self):
        """Test that the simulation context is built with the correct ground plane."""
        for add_ground_plane in (True, False):
            with self.subTest(add_ground_plane=add_ground_plane):
                with build_simulation_context(add_ground_plane=add_ground_plane) as _:
                    # Ensure that ground plane got added
                    self.assertEqual(is_prim_path_valid("/World/defaultGroundPlane"), add_ground_plane)

    def test_build_simulation_context_auto_add_lighting(self):
        """Test that the simulation context is built with the correct lighting."""
        for add_lighting in (True, False):
            for auto_add_lighting in (True, False):
                with self.subTest(add_lighting=add_lighting, auto_add_lighting=auto_add_lighting):
                    with build_simulation_context(add_lighting=add_lighting, auto_add_lighting=auto_add_lighting) as _:
                        if auto_add_lighting or add_lighting:
                            # Ensure that dome light got added
                            self.assertTrue(is_prim_path_valid("/World/defaultDomeLight"))
                        else:
                            # Ensure that dome light didn't get added
                            self.assertFalse(is_prim_path_valid("/World/defaultDomeLight"))

    def test_build_simulation_context_cfg(self):
        """Test that the simulation context is built with the correct cfg and values don't get overridden."""
        dt = 0.001
        # Non-standard gravity
        gravity = (0.0, 0.0, -1.81)
        device = "cuda:0"

        cfg = SimulationCfg(
            gravity=gravity,
            device=device,
            dt=dt,
        )

        with build_simulation_context(sim_cfg=cfg, gravity_enabled=False, dt=0.01, device="cpu") as sim:
            self.assertEqual(sim.cfg.gravity, gravity)
            self.assertEqual(sim.cfg.device, device)
            self.assertEqual(sim.cfg.dt, dt)


if __name__ == "__main__":
    run_tests()
