# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import isaaclab.utils.modifiers as modifiers
from isaaclab.test.utils import test_devices

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("cfg", "data", "expected"),
    [
        (modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0}), [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        (modifiers.ModifierCfg(func=modifiers.bias, params={"value": 1.0}), [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
        (modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (1.5, 2.5)}), [1.0, 2.0, 3.0], [1.5, 2.0, 2.5]),
        (modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (1.5, None)}), [1.0, 2.0, 3.0], [1.5, 2.0, 3.0]),
        (modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (None, 2.5)}), [1.0, 2.0, 3.0], [1.0, 2.0, 2.5]),
        (modifiers.ModifierCfg(func=torch.nn.functional.relu), [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
    ],
    ids=["scale", "bias", "clip", "clip_no_upper", "clip_no_lower", "relu"],
)
def test_stateless_modifiers(cfg, data, expected):
    output = cfg.func(torch.tensor(data), **cfg.params)
    torch.testing.assert_close(output, torch.tensor(expected))


@pytest.mark.parametrize("device", test_devices())
def test_digital_filter(device):
    """A first-order filter fed a sampled sine converges to the same value after every reset."""
    cfg = modifiers.DigitalFilterCfg(A=[0.0, 0.1], B=[0.5, 0.5])
    modifier = cfg.func(cfg, (3,), device=device)
    delta = torch.pi / torch.tensor([8.0, 8.0, 8.0], device=device)
    expected = torch.tensor([-0.45661893] * 3, device=device)

    for _ in range(2):
        modifier.reset()
        for i in range(16):
            processed = modifier(torch.sin(i * delta))
            assert processed.shape == (3,)
        torch.testing.assert_close(processed, expected)


@pytest.mark.parametrize("device", test_devices())
def test_integrator(device):
    """Trapezoidal integration of 0, 1, ..., 5 with dt=1 gives 12.5 and restarts from zero after reset."""
    cfg = modifiers.IntegratorCfg(dt=1.0)
    modifier = cfg.func(cfg, (1,), device=device)

    for _ in range(2):
        modifier.reset()
        for value in range(6):
            processed = modifier(torch.tensor([float(value)], device=device))
        torch.testing.assert_close(processed, torch.tensor([12.5], device=device))
