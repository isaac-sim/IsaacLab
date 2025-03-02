# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import unittest

import omni
from isaacsim.core.cloner import GridCloner

from isaaclab_assets import ANYMAL_D_CFG, CARTPOLE_CFG

from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.utils.timer import Timer


class TestRobotLoadPerformance(unittest.TestCase):
    """Test robot load performance."""

    """
    Tests
    """

    def test_robot_load_performance(self):
        """Test robot load time."""
        test_configs = {
            "Cartpole": {"robot_cfg": CARTPOLE_CFG, "expected_load_time": 10.0},
            "Anymal_D": {"robot_cfg": ANYMAL_D_CFG, "expected_load_time": 40.0},
        }
        for test_config in test_configs.items():
            for device in ("cuda:0", "cpu"):
                with self.subTest(test_config=test_config, device=device):
                    with build_simulation_context(device=device) as sim:
                        sim._app_control_on_stop_handle = None
                        cloner = GridCloner(spacing=2)
                        target_paths = cloner.generate_paths("/World/Robots", 4096)
                        omni.usd.get_context().get_stage().DefinePrim(target_paths[0], "Xform")
                        _ = cloner.clone(
                            source_prim_path=target_paths[0],
                            prim_paths=target_paths,
                            replicate_physics=False,
                            copy_from_source=True,
                        )
                        with Timer(f"{test_config[0]} load time for device {device}") as timer:
                            robot = Articulation(  # noqa: F841
                                test_config[1]["robot_cfg"].replace(prim_path="/World/Robots_.*/Robot")
                            )
                            sim.reset()
                            elapsed_time = timer.time_elapsed
                        self.assertLessEqual(elapsed_time, test_config[1]["expected_load_time"])


if __name__ == "__main__":
    run_tests()
