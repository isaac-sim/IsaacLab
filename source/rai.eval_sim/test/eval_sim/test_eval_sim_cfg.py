# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import tempfile
import unittest

from rai.eval_sim.eval_sim import EvalSimCfg


class TestEvalSimCfg(unittest.TestCase):
    def test_load_save_yaml(self):
        """Test that saving and loading yaml works."""
        cfg = EvalSimCfg()

        tmp = tempfile.NamedTemporaryFile()

        # create a temporary file to store the yaml
        with open(tmp.name, "w") as f:
            # write to yaml
            cfg.to_yaml(f.name)

        # verify that file exists, has 777 permissions and has content
        # file exists
        assert os.path.exists(tmp.name)

        with open(tmp.name) as f:
            # file has 007 permissions
            self.assertEqual(oct(os.stat(f.name).st_mode)[-3:], "007")

            # load the yaml
            cfg2 = EvalSimCfg.from_yaml(f.name)

        # verify that the loaded yaml is the same as the original
        self.assertEqual(cfg, cfg2)

    def test_str(self):
        """Test that the string representation of EvalSimCfg works."""
        cfg = EvalSimCfg()
        cfg.__str__()


if __name__ == "__main__":
    run_tests()
