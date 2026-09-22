# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import isaaclab.utils.noise as noise
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit

LOW = [0.1, 0.2, 0.3]
HIGH = [0.4, 0.5, 0.6]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
@pytest.mark.parametrize("noise_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("op", ["add", "scale", "abs"])
@pytest.mark.parametrize("kind", ["gaussian", "uniform", "constant"])
def test_noise_statistics_and_device_placement(device, noise_device, op, kind):
    """Noise parameters move to the data device and the applied noise matches the configured distribution."""
    # ones for 'scale' make the output equal to the noise term itself; torch.rand may return exact zeros
    data = torch.ones(2000, 3, device=device) if op == "scale" else torch.rand(2000, 3, device=device)
    low = torch.tensor(LOW, device=noise_device)
    high = torch.tensor(HIGH, device=noise_device)
    if kind == "gaussian":
        cfg = noise.GaussianNoiseCfg(std=low, mean=high, operation=op)
        params = ("std", "mean")
    elif kind == "uniform":
        cfg = noise.UniformNoiseCfg(n_min=low, n_max=high, operation=op)
        params = ("n_min", "n_max")
    else:
        cfg = noise.ConstantNoiseCfg(bias=low, operation=op)
        params = ("bias",)

    for _ in range(2):
        noisy = cfg.func(data, cfg=cfg)
        applied = noisy - data if op == "add" else noisy
        for param in params:
            assert str(getattr(cfg, param).device) == device
        if kind == "gaussian":
            std, mean = torch.std_mean(applied, dim=0)
            torch.testing.assert_close(std, cfg.std, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(mean, cfg.mean, atol=2e-2, rtol=2e-2)
        elif kind == "uniform":
            assert torch.all(applied >= cfg.n_min - 1e-5) and torch.all(applied <= cfg.n_max + 1e-5)
        else:
            torch.testing.assert_close(applied, cfg.bias.expand_as(applied))
