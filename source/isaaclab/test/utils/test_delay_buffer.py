# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import torch
import unittest
from collections.abc import Generator

from isaaclab.utils import DelayBuffer


class TestDelayBuffer(unittest.TestCase):
    """Test fixture for checking the delay buffer implementation."""

    def setUp(self):
        self.device: str = "cpu"
        self.batch_size: int = 10
        self.history_length: int = 4
        # create the buffer
        self.buffer = DelayBuffer(self.history_length, batch_size=self.batch_size, device=self.device)

    def test_constant_time_lags(self):
        """Test constant delay."""
        const_lag: int = 3

        self.buffer.set_time_lag(const_lag)

        all_data = []
        for i, data in enumerate(self._generate_data(20)):
            all_data.append(data)
            # apply delay
            delayed_data = self.buffer.compute(data)
            error = delayed_data - all_data[max(0, i - const_lag)]
            self.assertTrue(torch.all(error == 0))

    def test_reset(self):
        """Test resetting the last two batch indices after iteration `reset_itr`."""
        const_lag: int = 2
        reset_itr = 10

        self.buffer.set_time_lag(const_lag)

        all_data = []
        for i, data in enumerate(self._generate_data(20)):
            all_data.append(data)
            # from 'reset_itr' iteration reset the last and second-to-last environments
            if i == reset_itr:
                self.buffer.reset([-2, -1])
            # apply delay
            delayed_data = self.buffer.compute(data)
            # before 'reset_itr' is is similar to test_constant_time_lags
            # after that indices [-2, -1] should be treated separately
            if i < reset_itr:
                error = delayed_data - all_data[max(0, i - const_lag)]
                self.assertTrue(torch.all(error == 0))
            else:
                # error_regular = delayed_data[:-2] - all_data[max(0, i - const_lag)][:-2]
                error2_reset = delayed_data[-2, -1] - all_data[max(reset_itr, i - const_lag)][-2, -1]
                # self.assertTrue(torch.all(error_regular == 0))
                self.assertTrue(torch.all(error2_reset == 0))

    def test_random_time_lags(self):
        """Test random delays."""
        max_lag: int = 3
        time_lags = torch.randint(low=0, high=max_lag + 1, size=(self.batch_size,), dtype=torch.int, device=self.device)

        self.buffer.set_time_lag(time_lags)

        all_data = []
        for i, data in enumerate(self._generate_data(20)):
            all_data.append(data)
            # apply delay
            delayed_data = self.buffer.compute(data)
            true_delayed_index = torch.maximum(i - self.buffer.time_lags, torch.zeros_like(self.buffer.time_lags))
            true_delayed_index = true_delayed_index.tolist()
            for i in range(self.batch_size):
                error = delayed_data[i] - all_data[true_delayed_index[i]][i]
                self.assertTrue(torch.all(error == 0))

    """Helper functions."""

    def _generate_data(self, length: int) -> Generator[torch.Tensor]:
        """Data generator for testing the buffer."""
        for step in range(length):
            yield torch.full((self.batch_size, 1), step, dtype=torch.int, device=self.device)


if __name__ == "__main__":
    run_tests()
