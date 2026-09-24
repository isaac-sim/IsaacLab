# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Generator

import pytest
import torch

from isaaclab.test.utils import test_devices
from isaaclab.utils import DelayBuffer

pytestmark = pytest.mark.unit


@pytest.fixture(params=test_devices())
def delay_buffer(request):
    """Create a delay buffer for testing."""
    device: str = request.param
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


def test_random_time_lags(delay_buffer):
    """Test random delays."""
    max_lag: int = 3
    time_lags = torch.randint(
        low=0, high=max_lag + 1, size=(delay_buffer.batch_size,), dtype=torch.long, device=delay_buffer.device
    )

    # Indexed assignment must accept int64 lags as well as the buffer's int32 dtype.
    delay_buffer.set_time_lag(time_lags, list(range(delay_buffer.batch_size)))

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


def test_delay_buffer_cuda_graph(delay_buffer):
    """The write index advances on every replay, including across partial resets."""
    if not delay_buffer.device.startswith("cuda"):
        pytest.skip("CUDA graph replay requires CUDA.")
    with torch.cuda.device(delay_buffer.device):
        data = torch.zeros(delay_buffer.batch_size, 1, device=delay_buffer.device)
        delay_buffer.set_time_lag(2)
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
            expected = torch.full_like(data, max(0, step - 2))
            if step >= 4:
                expected[1] = max(4, step - 2)
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(read, expected)
