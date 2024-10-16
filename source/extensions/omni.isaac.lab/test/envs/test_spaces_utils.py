# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# Can set this to False to see the GUI for debugging
HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest
import numpy as np

from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple, Dict

from omni.isaac.lab.envs.utils import sample_space, spec_to_gym_space

class TestSpacesUtils(unittest.TestCase):
    """Test for spaces utils' functions"""

    """
    Tests
    """

    def test_spec_to_gym_space(self):
        # fundamental spaces
        # Box
        space = spec_to_gym_space(1)
        self.assertIsInstance(space, Box)
        self.assertEqual(space.shape, (1,))
        space = spec_to_gym_space([1, 2, 3, 4, 5])
        self.assertIsInstance(space, Box)
        self.assertEqual(space.shape, (1, 2, 3, 4, 5))
        space = spec_to_gym_space(Box(low=-1.0, high=1.0, shape=(1, 2)))
        self.assertIsInstance(space, Box)
        # Discrete
        space = spec_to_gym_space({2})
        self.assertIsInstance(space, Discrete)
        self.assertEqual(space.n, 2)
        space = spec_to_gym_space(Discrete(2))
        self.assertIsInstance(space, Discrete)
        # MultiDiscrete
        space = spec_to_gym_space([{1}, {2}, {3}])
        self.assertIsInstance(space, MultiDiscrete)
        self.assertEqual(space.nvec.shape, (3,))
        space = spec_to_gym_space(MultiDiscrete(np.array([1, 2, 3])))
        self.assertIsInstance(space, MultiDiscrete)
        # composite spaces
        # Tuple
        space = spec_to_gym_space(([1, 2, 3, 4, 5], {2}, [{1}, {2}, {3}]))
        self.assertIsInstance(space, Tuple)
        self.assertEqual(len(space), 3)
        self.assertIsInstance(space[0], Box)
        self.assertIsInstance(space[1], Discrete)
        self.assertIsInstance(space[2], MultiDiscrete)
        space = spec_to_gym_space(Tuple((Box(-1, 1, shape=(1,)), Discrete(2))))
        self.assertIsInstance(space, Tuple)
        # Dict
        space = spec_to_gym_space({"box": [1, 2, 3, 4, 5], "discrete": {2}, "multi_discrete": [{1}, {2}, {3}]})
        self.assertIsInstance(space, Dict)
        self.assertEqual(len(space), 3)
        self.assertIsInstance(space["box"], Box)
        self.assertIsInstance(space["discrete"], Discrete)
        self.assertIsInstance(space["multi_discrete"], MultiDiscrete)
        space = spec_to_gym_space(Dict({"box": Box(-1, 1, shape=(1,)), "discrete": Discrete(2)}))
        self.assertIsInstance(space, Dict)

    def test_sample_space(self):
        device = "cpu"
        # fundamental spaces
        # Box
        sample = sample_space(Box(low=-1.0, high=1.0, shape=(1, 2)), device, batch_size=1)
        self.assertIsInstance(sample, torch.Tensor)
        self._check_tensorized(sample, batch_size=1)
        # Discrete
        sample = sample_space(Discrete(2), device, batch_size=2)
        self.assertIsInstance(sample, torch.Tensor)
        self._check_tensorized(sample, batch_size=2)
        # MultiDiscrete
        sample = sample_space(MultiDiscrete(np.array([1, 2, 3])), device, batch_size=3)
        self.assertIsInstance(sample, torch.Tensor)
        self._check_tensorized(sample, batch_size=3)
        # composite spaces
        # Tuple
        sample = sample_space(Tuple((Box(-1, 1, shape=(1,)), Discrete(2))), device, batch_size=4)
        self.assertIsInstance(sample, (tuple, list))
        self._check_tensorized(sample, batch_size=4)
        # Dict
        sample = sample_space(Dict({"box": Box(-1, 1, shape=(1,)), "discrete": Discrete(2)}), device, batch_size=5)
        self.assertIsInstance(sample, dict)
        self._check_tensorized(sample, batch_size=5)

    """
    Helper functions.
    """

    def _check_tensorized(self, sample, batch_size):
        if isinstance(sample, (tuple, list)):
            list(map(self._check_tensorized, sample, [batch_size] * len(sample)))
        elif isinstance(sample, dict):
            list(map(self._check_tensorized, sample.values(), [batch_size] * len(sample)))
        else:
            self.assertIsInstance(sample, torch.Tensor)
            self.assertEqual(sample.shape[0], batch_size)

if __name__ == "__main__":
    run_tests()
