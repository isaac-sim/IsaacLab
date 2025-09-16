# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
from dataclasses import MISSING

import pytest

import isaaclab.utils.modifiers as modifiers
from isaaclab.utils import configclass


@configclass
class ModifierTestCfg:
    """Configuration for testing modifiers."""

    cfg: modifiers.ModifierCfg = MISSING
    init_data: torch.Tensor = MISSING
    result: torch.Tensor = MISSING
    num_iter: int = 10


def test_scale_modifier():
    """Test scale modifier."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    scale = 2.0
    result = torch.tensor([2.0, 4.0, 6.0])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": scale}),
        init_data=init_data,
        result=result,
    )

    # test modifier
    for _ in range(test_cfg.num_iter):
        output = test_cfg.cfg.func(test_cfg.init_data, **test_cfg.cfg.params)
        assert torch.allclose(output, test_cfg.result)


def test_bias_modifier():
    """Test bias modifier."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    bias = 1.0
    result = torch.tensor([2.0, 3.0, 4.0])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.ModifierCfg(func=modifiers.bias, params={"value": bias}),
        init_data=init_data,
        result=result,
    )

    # test modifier
    for _ in range(test_cfg.num_iter):
        output = test_cfg.cfg.func(test_cfg.init_data, **test_cfg.cfg.params)
        assert torch.allclose(output, test_cfg.result)


def test_clip_modifier():
    """Test clip modifier."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    min_val = 1.5
    max_val = 2.5
    result = torch.tensor([1.5, 2.0, 2.5])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (min_val, max_val)}),
        init_data=init_data,
        result=result,
    )

    # test modifier
    for _ in range(test_cfg.num_iter):
        output = test_cfg.cfg.func(test_cfg.init_data, **test_cfg.cfg.params)
        assert torch.allclose(output, test_cfg.result)


def test_clip_no_upper_bound_modifier():
    """Test clip modifier with no upper bound."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    min_val = 1.5
    result = torch.tensor([1.5, 2.0, 3.0])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (min_val, None)}),
        init_data=init_data,
        result=result,
    )

    # test modifier
    for _ in range(test_cfg.num_iter):
        output = test_cfg.cfg.func(test_cfg.init_data, **test_cfg.cfg.params)
        assert torch.allclose(output, test_cfg.result)


def test_clip_no_lower_bound_modifier():
    """Test clip modifier with no lower bound."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    max_val = 2.5
    result = torch.tensor([1.0, 2.0, 2.5])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (None, max_val)}),
        init_data=init_data,
        result=result,
    )

    # test modifier
    for _ in range(test_cfg.num_iter):
        output = test_cfg.cfg.func(test_cfg.init_data, **test_cfg.cfg.params)
        assert torch.allclose(output, test_cfg.result)


def test_torch_relu_modifier():
    """Test torch relu modifier."""
    # create test data
    init_data = torch.tensor([-1.0, 0.0, 1.0])
    result = torch.tensor([0.0, 0.0, 1.0])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.ModifierCfg(func=torch.nn.functional.relu),
        init_data=init_data,
        result=result,
    )

    # test modifier
    for _ in range(test_cfg.num_iter):
        output = test_cfg.cfg.func(test_cfg.init_data)
        assert torch.allclose(output, test_cfg.result)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_digital_filter(device):
    """Test digital filter modifier."""
    # create test data
    init_data = torch.tensor([0.0, 0.0, 0.0], device=device)
    A = [0.0, 0.1]
    B = [0.5, 0.5]
    result = torch.tensor([-0.45661893, -0.45661893, -0.45661893], device=device)

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.DigitalFilterCfg(A=A, B=B), init_data=init_data, result=result, num_iter=16
    )

    # create a modifier instance
    modifier_obj = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device=device)

    # test the modifier
    theta = torch.tensor([0.0], device=device)
    delta = torch.pi / torch.tensor([8.0, 8.0, 8.0], device=device)

    for _ in range(5):
        # reset the modifier
        modifier_obj.reset()

        # apply the modifier multiple times
        for i in range(test_cfg.num_iter):
            data = torch.sin(theta + i * delta)
            processed_data = modifier_obj(data)

            assert data.shape == processed_data.shape, "Modified data shape does not equal original"

        # check if the modified data is close to the expected result
        torch.testing.assert_close(processed_data, test_cfg.result)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_integral(device):
    """Test integral modifier."""
    # create test data
    init_data = torch.tensor([0.0], device=device)
    dt = 1.0
    result = torch.tensor([12.5], device=device)

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.IntegratorCfg(dt=dt),
        init_data=init_data,
        result=result,
        num_iter=6,
    )

    # create a modifier instance
    modifier_obj = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device=device)

    # test the modifier
    delta = torch.tensor(1.0, device=device)

    for _ in range(5):
        # reset the modifier
        modifier_obj.reset()

        # clone the data to avoid modifying the original
        data = test_cfg.init_data.clone()
        # apply the modifier multiple times
        for _ in range(test_cfg.num_iter):
            processed_data = modifier_obj(data)
            data = data + delta

            assert data.shape == processed_data.shape, "Modified data shape does not equal original"

        # check if the modified data is close to the expected result
        torch.testing.assert_close(processed_data, test_cfg.result)


def _counter_batch(t: int, shape, device):
    return torch.full(shape, float(t), device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_delayed_observation_fixed_lag(device):
    """Fixed lag (L=2) should return t-2 after warmup; shape preserved."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # config: fixed lag 2, single vector obs (3 envs)
    cfg = modifiers.DelayedObservationCfg(min_lag=2, max_lag=2, per_env=True, hold_prob=0.0, update_period=0)
    init_data = torch.zeros(3, device=device)  # shape carried into modifier ctor

    # choose iterations past warmup (max_lag+1 pushes) so last output reflects real history
    num_iter = cfg.max_lag + 6
    expected_final = torch.full_like(init_data, float((num_iter - 1) - 2))

    test_cfg = ModifierTestCfg(cfg=cfg, init_data=init_data, result=expected_final, num_iter=num_iter)

    # create a modifier instance
    modifier_obj = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device=device)

    for _ in range(3):  # a few trials with reset
        modifier_obj.reset()
        for t in range(test_cfg.num_iter):
            data = _counter_batch(t, test_cfg.init_data.shape, device)
            processed = modifier_obj(data)
            assert processed.shape == data.shape, "Modified data shape does not equal original"

        torch.testing.assert_close(processed, test_cfg.result)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_delayed_observation_multi_rate_period_3(device):
    """Multi-rate cadence: refresh every 3 steps with desired lag=2; holds in between."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # single env scalar obs; deterministic cadence (per_env_phase=False)
    cfg = modifiers.DelayedObservationCfg(
        min_lag=3, max_lag=3, per_env=True, hold_prob=0.0, update_period=3, per_env_phase=False
    )
    init_data = torch.zeros(1, device=device)

    num_iter = cfg.max_lag + 10

    # compute expected final value: last t minus realized lag under the 3-step cadence
    realized = None
    for t in range(num_iter):
        if realized is None:
            realized = 3
        elif ((t + 1) % cfg.update_period) == 0:  # refresh on every 3rd call
            realized = 3
        else:
            realized = min(realized + 1, cfg.max_lag)
    expected_final = torch.tensor([float((num_iter - 1) - realized)], device=device)

    test_cfg = ModifierTestCfg(cfg=cfg, init_data=init_data, result=expected_final, num_iter=num_iter)

    modifier_obj = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device=device)

    for _ in range(2):
        modifier_obj.reset()
        for t in range(test_cfg.num_iter):
            data = _counter_batch(t, test_cfg.init_data.shape, device)
            processed = modifier_obj(data)
            assert processed.shape == data.shape
        torch.testing.assert_close(processed, test_cfg.result)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_delayed_observation_bounds_and_causality(device):
    """Lag stays within [min_lag,max_lag] and obeys causal clamp: lag_t <= lag_{t-1}+1."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = modifiers.DelayedObservationCfg(min_lag=0, max_lag=4, per_env=True, hold_prob=0.0, update_period=0)
    init_data = torch.zeros(4, device=device)

    modifier_obj = cfg.func(cfg, init_data.shape, device=device)

    prev_lag = None
    num_iter = cfg.max_lag + 20
    for t in range(num_iter):
        out = modifier_obj(_counter_batch(t, init_data.shape, device))
        # infer realized lag from the counter signal: lag = t - out
        lag = (t - out).to(torch.long)

        if t >= (cfg.max_lag + 1):  # after warmup
            assert torch.all(lag >= cfg.min_lag) and torch.all(lag <= cfg.max_lag)
            if prev_lag is not None:
                assert torch.all(lag <= prev_lag + 1)
            prev_lag = lag