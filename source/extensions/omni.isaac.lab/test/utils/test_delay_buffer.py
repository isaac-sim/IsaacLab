# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import unittest

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app


from omni.isaac.lab.utils import DelayBuffer


class TestDelayBuffer(unittest.TestCase):
    """Test fixture for checking Delay Buffer utilities in Orbit."""

    device: str = "cpu"
    num_envs: int = 10
    max_num_histories: int = 4

    def generate_data(self, length: int) -> torch.Tensor:
        for step in range(length):
            yield torch.ones((self.num_envs, 1), dtype=int, device=self.device) * step

    def test_constant_time_lags(self):
        """Test constant delay."""
        const_lag: int = 3

        delay_buffer = DelayBuffer(self.max_num_histories, num_envs=self.num_envs, device=self.device)
        delay_buffer.set_time_lag(const_lag)

        all_data = []
        for i, data in enumerate(self.generate_data(20)):
            all_data.append(data)
            # apply delay
            delayed_data = delay_buffer.compute(data)
            error = delayed_data - all_data[max(0, i - const_lag)]
            self.assertTrue(torch.all(error == 0))

    def test_reset(self):
        """Test resetting the last two environments after iteration `reset_itr`."""
        const_lag: int = 2
        reset_itr = 10

        delay_buffer = DelayBuffer(self.max_num_histories, num_envs=self.num_envs, device=self.device)
        delay_buffer.set_time_lag(const_lag)

        all_data = []
        for i, data in enumerate(self.generate_data(20)):
            all_data.append(data)
            # from 'reset_itr' iteration reset the last and second-to-last environments
            if i == reset_itr:
                delay_buffer.reset([-2, -1])
            # apply delay
            delayed_data = delay_buffer.compute(data)
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
        """Test random delay."""
        max_lag: int = 3
        time_lags = torch.randint(low=0, high=max_lag + 1, size=(self.num_envs,), dtype=torch.int, device=self.device)

        delay_buffer = DelayBuffer(self.max_num_histories, num_envs=self.num_envs, device=self.device)
        delay_buffer.set_time_lag(time_lags)

        all_data = []
        for i, data in enumerate(self.generate_data(20)):
            all_data.append(data)
            # apply delay
            delayed_data = delay_buffer.compute(data)
            true_delayed_index = torch.maximum(i - delay_buffer.time_lags, torch.zeros_like(delay_buffer.time_lags))
            true_delayed_index = true_delayed_index.tolist()
            for i in range(self.num_envs):
                error = delayed_data[i] - all_data[true_delayed_index[i]][i]
                self.assertTrue(torch.all(error == 0))


if __name__ == "__main__":
    run_tests()
