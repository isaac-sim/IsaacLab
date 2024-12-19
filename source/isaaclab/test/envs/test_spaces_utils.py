# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# Can set this to False to see the GUI for debugging
HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import unittest
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

from isaaclab.envs.utils.spaces import deserialize_space, sample_space, serialize_space, spec_to_gym_space


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

    def test_space_serialization_deserialization(self):
        # fundamental spaces
        # Box
        space = 1
        output = deserialize_space(serialize_space(space))
        self.assertEqual(space, output)
        space = [1, 2, 3, 4, 5]
        output = deserialize_space(serialize_space(space))
        self.assertEqual(space, output)
        space = Box(low=-1.0, high=1.0, shape=(1, 2))
        output = deserialize_space(serialize_space(space))
        self.assertIsInstance(output, Box)
        self.assertTrue((space.low == output.low).all())
        self.assertTrue((space.high == output.high).all())
        self.assertEqual(space.shape, output.shape)
        # Discrete
        space = {2}
        output = deserialize_space(serialize_space(space))
        self.assertEqual(space, output)
        space = Discrete(2)
        output = deserialize_space(serialize_space(space))
        self.assertIsInstance(output, Discrete)
        self.assertEqual(space.n, output.n)
        # MultiDiscrete
        space = [{1}, {2}, {3}]
        output = deserialize_space(serialize_space(space))
        self.assertEqual(space, output)
        space = MultiDiscrete(np.array([1, 2, 3]))
        output = deserialize_space(serialize_space(space))
        self.assertIsInstance(output, MultiDiscrete)
        self.assertTrue((space.nvec == output.nvec).all())
        # composite spaces
        # Tuple
        space = ([1, 2, 3, 4, 5], {2}, [{1}, {2}, {3}])
        output = deserialize_space(serialize_space(space))
        self.assertEqual(space, output)
        space = Tuple((Box(-1, 1, shape=(1,)), Discrete(2)))
        output = deserialize_space(serialize_space(space))
        self.assertIsInstance(output, Tuple)
        self.assertEqual(len(output), 2)
        self.assertIsInstance(output[0], Box)
        self.assertIsInstance(output[1], Discrete)
        # Dict
        space = {"box": [1, 2, 3, 4, 5], "discrete": {2}, "multi_discrete": [{1}, {2}, {3}]}
        output = deserialize_space(serialize_space(space))
        self.assertEqual(space, output)
        space = Dict({"box": Box(-1, 1, shape=(1,)), "discrete": Discrete(2)})
        output = deserialize_space(serialize_space(space))
        self.assertIsInstance(output, Dict)
        self.assertEqual(len(output), 2)
        self.assertIsInstance(output["box"], Box)
        self.assertIsInstance(output["discrete"], Discrete)

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
