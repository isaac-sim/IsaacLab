# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Generator

import pytest
import torch

from isaaclab.test.utils import test_devices
from isaaclab.utils import DelayBuffer

pytestmark = pytest.mark.unit


def test_callable_compatibility_preserves_compute_overrides_and_super():
    """Mixed old and new implementations retain defining-class dispatch instead of recursing through self()."""

    with pytest.warns(DeprecationWarning, match="define __call__"):

        class Legacy(DelayBuffer):
            def compute(self, data):
                return super().compute(data + 1) * 2

    class Modern(Legacy):
        def __call__(self, data):
            return super().__call__(data + 3) * 4

    class LegacyMixin:
        def compute(self, data):
            return super().compute(data + 5) * 6

    with pytest.warns(DeprecationWarning, match="define __call__"):

        class Mixed(LegacyMixin, Modern):
            pass

    data = torch.tensor([[2.0]])
    buffer = Mixed(0, 1, "cpu")
    expected = ((data + 5 + 3 + 1) * 2) * 4 * 6
    with pytest.warns(DeprecationWarning, match="Use term"):
        torch.testing.assert_close(buffer(data), expected)
    with pytest.warns(DeprecationWarning, match="Use term"):
        torch.testing.assert_close(buffer.compute(data), expected)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.testing.assert_close(DelayBuffer(0, 1, "cpu")(data), data)
    assert not caught


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
        all_data.append(data)
        # apply delay
        delayed_data = delay_buffer(data)
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
        delayed_data = delay_buffer(data)
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
        delayed_data = delay_buffer(data)
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
    """``compute()``'s returned tensor must not alias the internal circular buffer storage.

    Regression: ``DelayBuffer.compute`` previously called ``.clone()`` defensively. After
    dropping the clone (advanced indexing returns a copy), this asserts the contract is
    preserved — mutating the result in place must not affect the next ``compute()`` output.
    """
    delay_buffer.set_time_lag(0)
    first = delay_buffer(torch.full((delay_buffer.batch_size, 1), 1, dtype=torch.int))
    first.fill_(999)  # mutate the returned tensor
    second = delay_buffer(torch.full((delay_buffer.batch_size, 1), 2, dtype=torch.int))
    assert torch.all(second == 2), "Mutation of a prior compute() result leaked into the next call"


def test_time_lag_updates(delay_buffer):
    """Subset updates accept integer tensors and empty selections, keeping extrema current."""
    delay_buffer.set_time_lag(2)
    delay_buffer.set_time_lag(torch.tensor([0, 4], dtype=torch.long), [0, 1])
    assert delay_buffer.min_time_lag == 0
    assert delay_buffer.max_time_lag == 4
    delay_buffer.set_time_lag(torch.empty(0, dtype=torch.int), [])
    delay_buffer.set_time_lag(1, [0, 1])
    assert delay_buffer.min_time_lag == 1
    assert delay_buffer.max_time_lag == 2
    # Bounded device-side sampling must not leave stale cached extrema.
    delay_buffer.time_lags.fill_(3)
    assert delay_buffer.min_time_lag == delay_buffer.max_time_lag == 3


@pytest.mark.parametrize("feature_shape", [(), (2, 3)])
def test_ring_storage_and_partial_reset(delay_buffer, feature_shape):
    """One slot changes per call; reset never exposes data from the previous episode."""
    import isaaclab.utils.buffers.delay_buffer as delay_module

    assert not hasattr(delay_module, "CircularBuffer"), "Delay storage must not depend on frame stacking."
    lags = list(range(5)) * 2
    delay_buffer.set_time_lag(torch.tensor(lags, device=delay_buffer.device))
    histories = [[] for _ in lags]
    for step in range(18):
        reset_ids = [0, 3, 8] if step == 7 else list(range(10)) if step == 12 else []
        storage = delay_buffer._buffer
        previous = storage.clone() if storage is not None else None
        delay_buffer.reset(reset_ids)
        for index in reset_ids:
            histories[index].clear()
        if storage is not None:
            torch.testing.assert_close(storage, previous)
        values = torch.arange(10, device=delay_buffer.device) + 100 * step
        data = values.view(10, *([1] * len(feature_shape))).expand(10, *feature_shape)
        result = delay_buffer(data)
        expected = []
        for index, history in enumerate(histories):
            history.append(100 * step + index)
            expected.append(history[max(0, len(history) - 1 - lags[index])])
        expected = torch.tensor(expected, device=delay_buffer.device).view(10, *([1] * len(feature_shape)))
        torch.testing.assert_close(result, expected.expand_as(data))
        if storage is not None:
            assert delay_buffer._buffer is storage
            for slot in range(5):
                if slot != step % 5:
                    torch.testing.assert_close(storage[slot], previous[slot])


def test_delay_buffer_cuda_graph(delay_buffer):
    """The write index advances on every replay, including across partial resets."""
    if not delay_buffer.device.startswith("cuda"):
        pytest.skip("CUDA graph replay requires CUDA.")
    with torch.cuda.device(delay_buffer.device):
        data = torch.zeros(delay_buffer.batch_size, 1, device=delay_buffer.device)
        delay_buffer.set_time_lag(2)
        delay_buffer(data)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = delay_buffer(data)
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
