# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import torch
from collections.abc import Generator

import pytest

from isaaclab.utils import DelayBuffer


@pytest.fixture
def delay_buffer():
    """Create a delay buffer for testing."""
    device: str = "cpu"
    batch_size: int = 10
    history_length: int = 4
    return DelayBuffer(history_length, batch_size=batch_size, device=device)


def _generate_data(batch_size: int, length: int, device: str) -> Generator[torch.Tensor]:
    """Data generator for testing the buffer."""
    for step in range(length):
        yield torch.full((batch_size, 1), step, dtype=torch.int, device=device)


def test_constant_time_lags(delay_buffer):
    """Test constant delay."""
    const_lag: int = 3
    batch_size: int = 10

    delay_buffer.set_time_lag(const_lag)

    all_data = []
    for i, data in enumerate(_generate_data(batch_size, 20, delay_buffer.device)):
        all_data.append(data)
        # apply delay
        delayed_data = delay_buffer.compute(data)
        error = delayed_data - all_data[max(0, i - const_lag)]
        assert torch.all(error == 0)


def test_reset(delay_buffer):
    """Test resetting the last two batch indices after iteration `reset_itr`."""
    const_lag: int = 2
    reset_itr = 10
    batch_size: int = 10

    delay_buffer.set_time_lag(const_lag)

    all_data = []
    for i, data in enumerate(_generate_data(batch_size, 20, delay_buffer.device)):
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
            assert torch.all(error == 0)
        else:
            # error_regular = delayed_data[:-2] - all_data[max(0, i - const_lag)][:-2]
            error2_reset = delayed_data[-2, -1] - all_data[max(reset_itr, i - const_lag)][-2, -1]
            # assert torch.all(error_regular == 0)
            assert torch.all(error2_reset == 0)


def test_random_time_lags(delay_buffer):
    """Test random delays."""
    max_lag: int = 3
    time_lags = torch.randint(
        low=0, high=max_lag + 1, size=(delay_buffer.batch_size,), dtype=torch.int, device=delay_buffer.device
    )

    delay_buffer.set_time_lag(time_lags)

    all_data = []
    for i, data in enumerate(_generate_data(delay_buffer.batch_size, 20, delay_buffer.device)):
        all_data.append(data)
        # apply delay
        delayed_data = delay_buffer.compute(data)
        true_delayed_index = torch.maximum(i - delay_buffer.time_lags, torch.zeros_like(delay_buffer.time_lags))
        true_delayed_index = true_delayed_index.tolist()
        for i in range(delay_buffer.batch_size):
            error = delayed_data[i] - all_data[true_delayed_index[i]][i]
            assert torch.all(error == 0)
