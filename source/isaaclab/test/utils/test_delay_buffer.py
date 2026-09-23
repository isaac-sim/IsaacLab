# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Generator

import pytest
import torch

from isaaclab.test.utils import test_devices
from isaaclab.utils import DelayBuffer
from isaaclab.utils.delay import DelayCfg, _Delay

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


def test_compute_result_independent_of_internal_buffer(delay_buffer):
    """``compute()``'s returned tensor must not alias the internal circular buffer storage.

    Regression: ``DelayBuffer.compute`` previously called ``.clone()`` defensively. After
    dropping the clone (advanced indexing returns a copy), this asserts the contract is
    preserved — mutating the result in place must not affect the next ``compute()`` output.
    """
    delay_buffer.set_time_lag(0)
    first = delay_buffer.compute(torch.full((delay_buffer.batch_size, 1), 1, dtype=torch.int))
    first.fill_(999)  # mutate the returned tensor
    second = delay_buffer.compute(torch.full((delay_buffer.batch_size, 1), 2, dtype=torch.int))
    assert torch.all(second == 2), "Mutation of a prior compute() result leaked into the next call"


@pytest.mark.parametrize("time_lag", [-1, 5, 2**32, torch.tensor([-1, 2]), torch.tensor([5, 2])])
def test_invalid_time_lag_does_not_mutate_state(delay_buffer, time_lag, monkeypatch):
    """Validate external lags before writing or cloning the live configuration."""
    lags = delay_buffer.time_lags
    delay_buffer.set_time_lag(2)
    monkeypatch.setattr(lags, "clone", lambda: pytest.fail("Do not clone live lag state to validate inputs."))
    with pytest.raises(ValueError):
        delay_buffer.set_time_lag(time_lag, [0, 1])
    assert delay_buffer.time_lags is lags
    assert torch.all(lags == 2)
    assert delay_buffer.min_time_lag == delay_buffer.max_time_lag == 2
    for step in range(6):
        result = delay_buffer.compute(torch.full((delay_buffer.batch_size,), step, device=delay_buffer.device))
        assert torch.all(result == max(0, step - 2))


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
        result = delay_buffer.compute(data)
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
        delay_buffer.compute(data)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = delay_buffer.compute(data)
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


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("shape", [(3,), (3, 2, 2)])
@pytest.mark.parametrize("settings", [{"min_lag": 2, "max_lag": 2}, {"update_period": 3}, {"hold_prob": 1.0}])
def test_delay_delivery_and_reset(device, shape, settings):
    """Fixed latency, sensor cadence, and holds share shape-preserving reset semantics."""
    cfg = DelayCfg(term=None, per_env_phase=False, **settings)
    delay = _Delay(cfg, shape[0], device)
    for step in range(12):
        if step == 5:
            delay.reset([1])
        data = torch.full(shape, step + 10, dtype=torch.float64, device=device)
        result = delay(data)
        expected = []
        for env in range(shape[0]):
            start = 5 if env == 1 and step >= 5 else 0
            if cfg.hold_prob == 1.0:
                sample = start
            elif cfg.update_period > 1:
                sample = start + (step - start) // cfg.update_period * cfg.update_period
            else:
                sample = max(start, step - cfg.max_lag)
            expected.append(sample + 10)
        expected = torch.tensor(expected, dtype=data.dtype, device=device).view(3, *([1] * (len(shape) - 1)))
        torch.testing.assert_close(result, expected.expand_as(data))
        result.fill_(-999)  # Downstream in-place processing must not corrupt held frames.


@pytest.mark.parametrize("device", test_devices())
def test_delay_stochastic_delivery(device):
    """Jitter never delivers an older sample, and bounded GPU lag generation never calls the setter."""
    cfg = DelayCfg(term=None, min_lag=1, max_lag=4, update_period=3, hold_prob=0.2)
    delay = _Delay(cfg, 16, device)
    assert "_step" not in vars(delay), "The buffer owns the per-environment clock."
    delay._buffer.set_time_lag = lambda *args: pytest.fail("The hot path must not validate lags on the host.")
    previous = torch.full((16, 1), -1.0, device=device)
    for step in range(40):
        output = delay(torch.full_like(previous, step))
        assert torch.all(output >= previous)
        assert torch.all(output <= max(0, step - cfg.min_lag))
        assert torch.all(delay._buffer.time_lags >= cfg.min_lag)
        assert torch.all(delay._buffer.time_lags <= cfg.max_lag)
        previous = output

    shared = _Delay(DelayCfg(term=None, max_lag=4, per_env=False), 16, device)
    for step in range(12):
        output = shared(torch.full_like(previous, step))
        assert torch.all(output == output[0])


@pytest.mark.parametrize(
    "settings", [{"min_lag": -1}, {"min_lag": 2, "max_lag": 1}, {"update_period": 0}, {"hold_prob": 1.1}]
)
def test_delay_cfg_validation(settings):
    """Reject invalid scheduling parameters in the config before allocating any buffer."""
    cfg = DelayCfg(term=None, **settings)
    with pytest.raises(ValueError):
        cfg.validate()
    with pytest.raises(ValueError):
        _Delay(cfg, 2, "cpu")


@pytest.mark.parametrize("device", test_devices())
def test_delay_schedule_cuda_graph(device):
    """Cadence and held output advance on-device during graph replay."""
    if not device.startswith("cuda"):
        pytest.skip("CUDA graph replay requires CUDA.")
    with torch.cuda.device(device):
        delay = _Delay(DelayCfg(term=None, update_period=3, per_env_phase=False), 2, device)
        data = torch.zeros(2, 1, device=device)
        delay(data)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = delay(data)
        delay.reset()
        for step in range(10):
            if step == 5:
                delay.reset([1])
            data.fill_(step)
            graph.replay()
            expected = torch.full_like(data, step // 3 * 3)
            if step >= 5:
                expected[1] = 5 + (step - 5) // 3 * 3
            torch.testing.assert_close(output, expected)
