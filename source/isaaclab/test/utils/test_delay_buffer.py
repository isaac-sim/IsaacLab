# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils import DelayBuffer

pytestmark = pytest.mark.unit

BATCH_SIZE = 10
NUM_STEPS = 8


@pytest.fixture
def delay_buffer():
    return DelayBuffer(history_length=4, batch_size=BATCH_SIZE, device="cpu")


def step_data(step: int) -> torch.Tensor:
    return torch.full((BATCH_SIZE, 1), step, dtype=torch.int)


def test_constant_time_lag(delay_buffer):
    delay_buffer.set_time_lag(3)
    for step in range(NUM_STEPS):
        delayed = delay_buffer.compute(step_data(step))
        torch.testing.assert_close(delayed, step_data(max(0, step - 3)))


def test_per_batch_time_lags(delay_buffer):
    time_lags = torch.randint(0, 4, (BATCH_SIZE,), dtype=torch.int)
    delay_buffer.set_time_lag(time_lags)
    for step in range(NUM_STEPS):
        delayed = delay_buffer.compute(step_data(step))
        expected = torch.clamp(step - time_lags, min=0).to(torch.int).unsqueeze(-1)
        torch.testing.assert_close(delayed, expected)


def test_reset_restarts_history_for_selected_batches(delay_buffer):
    lag, reset_step = 2, 4
    delay_buffer.set_time_lag(lag)
    for step in range(NUM_STEPS):
        if step == reset_step:
            delay_buffer.reset([-2, -1])
        delayed = delay_buffer.compute(step_data(step))
        expected = step_data(max(0, step - lag))
        if step >= reset_step:
            expected[-2:] = max(reset_step, step - lag)
        torch.testing.assert_close(delayed, expected)


@pytest.mark.parametrize(
    ("time_lag", "batch_ids"),
    [(5, [2]), (-1, [2]), (torch.tensor([5, 1], dtype=torch.int), [2, 3])],
)
def test_invalid_time_lag_does_not_mutate_state(delay_buffer, time_lag, batch_ids):
    """Invalid lags are rejected before the live lag configuration changes."""
    initial_lags = torch.arange(BATCH_SIZE, dtype=torch.int) % 5
    delay_buffer.set_time_lag(initial_lags)
    with pytest.raises(ValueError):
        delay_buffer.set_time_lag(time_lag, batch_ids)
    torch.testing.assert_close(delay_buffer.time_lags, initial_lags)
    assert (delay_buffer.min_time_lag, delay_buffer.max_time_lag) == (0, 4)


def test_compute_result_does_not_alias_internal_storage(delay_buffer):
    """Mutating a returned tensor in place must not leak into the next compute() output."""
    delay_buffer.set_time_lag(0)
    delay_buffer.compute(step_data(1)).fill_(999)
    torch.testing.assert_close(delay_buffer.compute(step_data(2)), step_data(2))
