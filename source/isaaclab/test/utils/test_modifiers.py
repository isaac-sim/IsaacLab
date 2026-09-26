# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import pytest
import torch

import isaaclab.utils.modifiers as modifiers
from isaaclab.test.utils import test_devices
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit


@configclass
class ModifierTestCfg:
    """Configuration for testing modifiers."""

    cfg: modifiers.ModifierCfg = MISSING
    init_data: torch.Tensor = MISSING
    result: torch.Tensor = MISSING
    num_iter: int = 10


@pytest.mark.parametrize(
    "func, params, result",
    [
        (modifiers.scale, {"multiplier": 2.0}, [2.0, 4.0, 6.0]),
        (modifiers.bias, {"value": 1.0}, [2.0, 3.0, 4.0]),
        (modifiers.clip, {"bounds": (1.5, 2.5)}, [1.5, 2.0, 2.5]),
        (modifiers.clip, {"bounds": (1.5, None)}, [1.5, 2.0, 3.0]),
        (modifiers.clip, {"bounds": (None, 2.5)}, [1.0, 2.0, 2.5]),
    ],
    ids=["scale", "bias", "clip", "clip_no_upper_bound", "clip_no_lower_bound"],
)
def test_stateless_modifiers(func, params, result):
    """Test the stateless scale, bias, and clip modifiers."""
    cfg = modifiers.ModifierCfg(func=func, params=params)
    output = cfg.func(torch.tensor([1.0, 2.0, 3.0]), **cfg.params)
    assert torch.allclose(output, torch.tensor(result))


@pytest.mark.parametrize("device", test_devices())
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


@pytest.mark.parametrize("device", test_devices())
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
