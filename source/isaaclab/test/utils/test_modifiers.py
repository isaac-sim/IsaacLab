# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
if not AppLauncher.instance():
    simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import pytest
from dataclasses import MISSING

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


def test_digital_filter():
    """Test digital filter modifier."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    A = [0.0, 0.1]
    B = [0.5, 0.5]
    result = torch.tensor([-0.45661893, -0.45661893, -0.45661893])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.DigitalFilterCfg(A=A, B=B),
        init_data=init_data,
        result=result,
    )

    # test modifier
    modifier_obj = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device="cpu")
    output = modifier_obj(test_cfg.init_data)
    assert torch.allclose(output, test_cfg.result)

    # test modifier with history
    for _ in range(test_cfg.num_iter):
        modifier_obj.reset()
        output = modifier_obj(test_cfg.init_data)
        assert torch.allclose(output, test_cfg.result)


def test_integral():
    """Test integral modifier."""
    # create test data
    init_data = torch.tensor([1.0, 2.0, 3.0])
    dt = 0.1
    result = torch.tensor([0.1, 0.3, 0.6])

    # create test config
    test_cfg = ModifierTestCfg(
        cfg=modifiers.IntegratorCfg(dt=dt),
        init_data=init_data,
        result=result,
    )

    # test modifier
    modifier_obj = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape, device="cpu")
    output = modifier_obj(test_cfg.init_data)
    assert torch.allclose(output, test_cfg.result)

    # test modifier with history
    for _ in range(test_cfg.num_iter):
        modifier_obj.reset()
        output = modifier_obj(test_cfg.init_data)
        assert torch.allclose(output, test_cfg.result)


if __name__ == "__main__":
    run_tests()
