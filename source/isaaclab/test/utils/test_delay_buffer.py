# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Generator

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import DelayBuffer

pytestmark = pytest.mark.unit


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
        # Reads before the first recorded sample use the input without allocating history.
        if i == 0:
            torch.testing.assert_close(delay_buffer.compute(data, update_history=False), data)
            assert torch.all(delay_buffer.num_pushes == 0)
        all_data.append(data)
        # apply delay
        delayed_data = delay_buffer.compute(data)
        error = delayed_data - all_data[max(0, i - const_lag)]
        assert torch.all(error == 0)
        torch.testing.assert_close(delay_buffer.compute(data + 100, update_history=False), delayed_data)
        assert torch.all(delay_buffer.num_pushes == i + 1)


@pytest.mark.parametrize("feature_shape", [(), (2, 3)])
def test_reset(delay_buffer, feature_shape):
    """Partial and full resets return fresh samples without affecting other histories."""
    import isaaclab.utils.buffers.delay_buffer as delay_module

    assert not hasattr(delay_module, "CircularBuffer"), "Delay storage must not depend on frame stacking."
    delay_buffer.set_time_lag(2)
    first_step = torch.zeros(delay_buffer.batch_size, dtype=torch.long, device=delay_buffer.device)
    shape = (delay_buffer.batch_size, *([1] * len(feature_shape)))
    for step in range(20):
        data = torch.full((delay_buffer.batch_size, *feature_shape), step, device=delay_buffer.device)
        if step in (7, 12):
            ids = [1] if step == 7 else None
            delay_buffer.reset(ids)
            first_step[ids if ids is not None else slice(None)] = step
            expected = torch.maximum(first_step, torch.full_like(first_step, step - 3))
            torch.testing.assert_close(
                delay_buffer.compute(data, update_history=False), expected.view(shape).expand_as(data)
            )
        expected = torch.maximum(first_step, torch.full_like(first_step, step - 2))
        torch.testing.assert_close(delay_buffer.compute(data), expected.view(shape).expand_as(data))


@pytest.mark.parametrize("hold_prob", [0.0, 0.5, 1.0])
def test_random_time_lags(delay_buffer, hold_prob, monkeypatch):
    """Each batch retains its lag or resamples; reset always starts a new episode."""
    delay_buffer = DelayBuffer(4, delay_buffer.batch_size, delay_buffer.device, min_lag=1, hold_prob=hold_prob)
    time_lags = torch.randint(1, 5, (delay_buffer.batch_size,), device=delay_buffer.device)
    # Indexed assignment must accept int64 lags as well as the buffer's int32 dtype.
    delay_buffer.set_time_lag(time_lags, list(range(delay_buffer.batch_size)))
    expected_lags = time_lags.int()
    first_step = torch.zeros_like(expected_lags)
    draws = torch.tensor([0.25, 0.75] * (delay_buffer.batch_size // 2), device=delay_buffer.device)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: draws)
    monkeypatch.setattr(torch, "randint", lambda low, high, size, **kwargs: torch.full(size, sampled_lag, **kwargs))

    for step, data in enumerate(_generate_data(delay_buffer.batch_size, 12, delay_buffer.device)):
        sampled_lag = 1 + step % 4
        if step == 7:
            delay_buffer.reset([0])
            expected_lags[0] = sampled_lag
            first_step[0] = step
        expected_lags[draws >= hold_prob] = sampled_lag
        result = delay_buffer.compute(data)
        torch.testing.assert_close(delay_buffer.time_lags, expected_lags)
        expected = torch.maximum(step - expected_lags, first_step).unsqueeze(-1)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(delay_buffer.compute(data + 100, update_history=False), expected)


@pytest.mark.parametrize(
    ("time_lag", "batch_ids"),
    [
        (5, [2]),
        (-1, [2]),
        (torch.tensor([5, 1], dtype=torch.int), [2, 3]),
    ],
)
def test_invalid_time_lag_does_not_mutate_state(delay_buffer, time_lag, batch_ids, monkeypatch):
    """Reject invalid inputs before copying or changing the live lag configuration."""
    initial_lags = torch.arange(delay_buffer.batch_size, dtype=torch.int) % 5
    delay_buffer.set_time_lag(initial_lags)
    expected_lags = delay_buffer.time_lags
    expected_values = expected_lags.clone()
    monkeypatch.setattr(expected_lags, "clone", lambda: pytest.fail("Validate the requested lag before copying state."))

    with pytest.raises(ValueError):
        delay_buffer.set_time_lag(time_lag, batch_ids)

    assert delay_buffer.time_lags is expected_lags
    assert torch.equal(delay_buffer.time_lags, expected_values)
    assert delay_buffer.min_time_lag == 0
    assert delay_buffer.max_time_lag == 4


def test_compute_result_independent_of_internal_buffer(delay_buffer):
    """Mutating a delayed output must not corrupt retained samples."""
    delay_buffer.set_time_lag(1)
    first = delay_buffer.compute(torch.full((delay_buffer.batch_size, 1), 1, dtype=torch.int))
    first.fill_(999)
    second = delay_buffer.compute(torch.full((delay_buffer.batch_size, 1), 2, dtype=torch.int))
    assert torch.all(second == 1)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_delay_buffer_cuda_graph(delay_buffer, device):
    """Sampling and ring writes work during graph replay, including across partial resets."""
    with torch.cuda.device(device):
        delay_buffer = DelayBuffer(4, delay_buffer.batch_size, device, min_lag=1, hold_prob=0.5)
        data = torch.zeros(delay_buffer.batch_size, 1, device=delay_buffer.device)
        delay_buffer.compute(data)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = delay_buffer.compute(data)
            read = delay_buffer.compute(data, update_history=False)
        delay_buffer.reset()
        for step in range(9):
            if step == 4:
                delay_buffer.reset([1])
            data.fill_(step)
            graph.replay()
            expected = (step - delay_buffer.time_lags).clamp_min(0).unsqueeze(-1).to(data.dtype)
            if step >= 4:
                expected[1].clamp_(min=4)
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(read, expected)
