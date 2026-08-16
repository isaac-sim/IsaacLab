# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.noise.noise_model import NoiseModelWithAdditiveBias

pytestmark = pytest.mark.unit


def _identity_noise(data, cfg):
    return data


def _unit_bias(data, cfg):
    return torch.ones_like(data)


def _indexed_bias(data, cfg):
    return torch.arange(data.numel(), device=data.device, dtype=data.dtype).reshape_as(data)


@pytest.mark.parametrize("sample_bias_per_component", [False, True])
def test_additive_bias_broadcasts_over_multidimensional_data(sample_bias_per_component):
    cfg = SimpleNamespace(
        noise_cfg=SimpleNamespace(func=_identity_noise),
        bias_noise_cfg=SimpleNamespace(func=_unit_bias),
        sample_bias_per_component=sample_bias_per_component,
    )
    model = NoiseModelWithAdditiveBias(cfg, num_envs=2, device="cpu")
    model.reset()

    data = torch.zeros((2, 3, 4, 5))
    output = model(data)

    assert output.shape == data.shape
    torch.testing.assert_close(output, torch.ones_like(data))


def test_additive_bias_samples_each_multidimensional_component_independently():
    cfg = SimpleNamespace(
        noise_cfg=SimpleNamespace(func=_identity_noise),
        bias_noise_cfg=SimpleNamespace(func=_indexed_bias),
        sample_bias_per_component=True,
    )
    model = NoiseModelWithAdditiveBias(cfg, num_envs=2, device="cpu")
    model.reset()

    data = torch.zeros((2, 3, 4, 5))
    output = model(data)
    expected_bias = torch.arange(data.numel(), dtype=data.dtype).reshape_as(data)

    assert model._bias.shape == data.shape
    torch.testing.assert_close(output, expected_bias)